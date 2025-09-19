# router_agent.py

import os
import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from gemini_utils import generate_explanation_with_gemini


# Load your existing recommendation models
from recommendation import recommend_products_for_user


# --- RAG Setup ---
def get_vector_db(embeddings):
    """Initializes or loads the vector database for RAG."""
    if os.path.exists("insurance_db"):
        return Chroma(persist_directory="insurance_db", embedding_function=embeddings)
    
    loader = PyPDFLoader("data/Insurance_Product_Catalog.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Fast batch processing - no sleep needed
    vector_db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="insurance_db")
    return vector_db

def get_translation_chain(llm):
    """Creates a chain to translate a user's query to English."""
    translation_prompt = PromptTemplate.from_template(
        "Translate the following text to standard English. If it is already in English, return it as is. Do not add any extra text or commentary.\n\nText: {text}"
    )
    return translation_prompt | llm | StrOutputParser()

def get_router_chain(llm):
    """Creates a classification chain to determine user intent."""
    router_prompt_template = """
    Based on the user's query and the LAST_INTENT of the conversation, classify the current intent into one of the following categories:
    - "recommendation_start": if the user is asking for a plan recommendation. This intent is also triggered when the user provides personal details such as their job, income, or number of dependents.
    - "recommendation_followup": if the user is providing details after being asked for them in the LAST_INTENT. This intent should only be used if the LAST_INTENT was "recommendation_start".
    - "follow_up_recommendation": if the user is asking a follow-up question about a previously recommended product, such as asking for an explanation, more details, or pricing.
    - "general_question": for all other questions about insurance policies, claims, or general information.
    
    Respond with ONLY the category name. Do not include any other text.
    LAST_INTENT: {last_intent}
    Query: {query}
    """
    router_prompt = PromptTemplate.from_template(router_prompt_template)
    return router_prompt | llm | StrOutputParser()

def get_rag_chain(llm, vector_db):
    """Creates the RAG chain for general questions."""
    retriever = vector_db.as_retriever()
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context. Context: {context}"),
        ("human", "{input}"),
    ])
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, rag_prompt))

def get_data_extraction_chain(llm, known_jobs, known_regions):
    """
    Creates a chain to extract and normalize structured data from user input.
    """
    job_labels = ", ".join(known_jobs)
    region_labels = ", ".join(known_regions)

    rec_prompt_template = f"""
    Extract the following information from the user's message and provide a JSON output.
    The keys should be exactly "Job", "Region", "Monthly_Income", and "Number_of_Dependents".
    Pay close attention to numerical values even if they are written informally. For example, "30000 naira" should be extracted as 30000.

    For "Job", find the closest matching job from this list: {job_labels}.
    For "Region", find the closest matching region from this list: {region_labels}.

    If a piece of information is missing, leave the value as "null".
    Only provide the JSON output. Do not add any extra text.

    User: {{input}}
    """
    rec_prompt = ChatPromptTemplate.from_template(rec_prompt_template)
    return rec_prompt | llm | JsonOutputParser()

def get_reverse_translation_chain(llm, target_language):
    """Creates a chain to translate a text back to the user's language."""
    translation_prompt = PromptTemplate.from_template(
        f"Translate the following text to {target_language}. Do not add any extra text or commentary.\n\nText: {{text}}"
    )
    return translation_prompt | llm | StrOutputParser()


def process_user_query(user_query, llm, vector_db, recommendation_function, recommendation_models, session_state):
    """Routes the user's query to the correct handler."""
    
    language = session_state.get("language", "English")
    translation_chain = get_translation_chain(llm)
    translated_query = translation_chain.invoke({"text": user_query})
    
    last_intent = session_state.get("last_intent", "none")
    router = get_router_chain(llm)
    intent = router.invoke({"query": translated_query, "last_intent": last_intent})

    if "follow_up_recommendation" in intent.lower():
        if session_state.get("last_recommendation") and session_state.get("chat_profile"):
            last_recommendation = session_state["last_recommendation"]
            user_profile = session_state["chat_profile"]
            
            explanation = generate_explanation_with_gemini(user_profile, last_recommendation, language)
            return explanation, "follow_up_recommendation"
        else:
            return ("I don't have a recent recommendation to explain. Could you please provide your details again?", language), "follow_up_recommendation"

    elif "recommendation" in intent.lower():
        known_jobs = list(recommendation_models["le_job"].classes_)
        known_regions = list(recommendation_models["le_region"].classes_)
        data_extraction_chain = get_data_extraction_chain(llm, known_jobs, known_regions)

        try:
            parsed_profile = data_extraction_chain.invoke({"input": translated_query})

            required_keys = ["Job", "Region", "Monthly_Income", "Number_of_Dependents"]
            missing_keys = [key for key in required_keys if not parsed_profile.get(key)]
            
            if missing_keys:
                missing_str = ", ".join(missing_keys).replace("_", " ")
                
                english_response = f"I need a little more information to give you a recommendation. Can you provide your {missing_str}?"
                
                if language != "English":
                    reverse_translation_chain = get_reverse_translation_chain(llm, language)
                    pidgin_response = reverse_translation_chain.invoke({"text": english_response})
                    session_state["last_intent"] = "recommendation_start"
                    return pidgin_response, "recommendation_start"
                else:
                    session_state["last_intent"] = "recommendation_start"
                    return english_response, "recommendation_start"

            monthly_income = parsed_profile.get('Monthly_Income')
            if monthly_income is not None:
                monthly_income = float(monthly_income)
            
            user_profile = {
                'Job': parsed_profile.get('Job'),
                'Region': parsed_profile.get('Region'),
                'Monthly_Income': monthly_income,
                'Number_of_Dependents': parsed_profile.get('Number_of_Dependents'),
                'Max_Affordable_Premium': monthly_income * 0.1
            }

            recs = recommendation_function(
                user_profile,
                recommendation_models["product_df"],
                recommendation_models["kmeans_model"],
                recommendation_models["scaler"],
                recommendation_models["le_job"],
                recommendation_models["le_region"],
                recommendation_models["cluster_product_map"]   ,
                recommendation_models["job_keywords"]
                             
            )

            if recs:
                session_state["chat_profile"] = user_profile
                session_state["last_recommendation"] = recs[0]
                
                # Build the full English response first ---
                english_rec_str = "Based on your information, here are some recommended plans:" + "\n\n"
                for rec in recs:
                    english_rec_str += f"**- {rec['Product_Name']}**\n"
                    english_rec_str += "**Monthly Premium:**" + f" ₦{rec['Monthly_Premium']:,.2f}\n"
                    english_rec_str += "**Reasoning:**" + f" {', '.join(rec['Reasons'])}\n\n"
                    explanation = generate_explanation_with_gemini(user_profile, rec, language)
                    english_rec_str += f"**** {explanation}\n\n"
                    
                english_rec_str += "Is there any product you would like to buy or get more details about?\n\n"

                # Translate the entire response to Pidgin if needed ---
                if language != "English":
                    reverse_translation_chain = get_reverse_translation_chain(llm, language)
                    pidgin_rec_str = reverse_translation_chain.invoke({"text": english_rec_str})
                    return pidgin_rec_str, "recommendation_success"
                else:
                    return english_rec_str, "recommendation_success"
            else:
                return ("I'm sorry, I couldn't find a suitable recommendation for you based on the information provided.", language), "recommendation_failure"
            
        except ValueError:
            return ("There was an error processing your income. Please make sure it's a number (e.g., 40000).", language), "data_error"
        except Exception as e:
            return (f"There was an error processing your recommendation request: {e}", language), "recommendation_error"

    elif "general_question" in intent.lower():
        rag_chain = get_rag_chain(llm, vector_db)
        response = rag_chain.invoke({"input": translated_query})

        if language != "English":
            reverse_translation_chain = get_reverse_translation_chain(llm, language)
            final_response = reverse_translation_chain.invoke({"text": response['answer']})
            return final_response, "general_question"
        
        return response['answer'], "general_question"

    return ("I'm sorry, I couldn't understand your request. Can you please rephrase it?", language), "unknown"
