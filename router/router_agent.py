# router_handler.py

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
    loader = PyPDFLoader("data/insurance_policy.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="insurance_db")
    return vector_db

# --- Chains ---
def get_router_chain(llm):
    """Creates a classification chain to determine user intent."""
    router_prompt_template = """
    Based on the user's query, classify the intent into one of the following categories:
    - "recommendation": if the user is asking for a plan recommendation. This intent is also triggered when the user provides personal details such as their job, income, or number of dependents, as this is an implicit request for a recommendation.
    - "follow_up_recommendation": if the user is asking a follow-up question about a previously recommended product, such as asking for an explanation, more details, or pricing.
    - "general_question": for all other questions about insurance policies, claims, or general information.
    
    Respond with ONLY the category name. Do not include any other text.
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

    For "Job", find the closest matching job from this list: {job_labels}.
    For "Region", find the closest matching region from this list: {region_labels}.

    If a piece of information is missing, leave the value as "null".
    Only provide the JSON output. Do not add any extra text.

    User: {{input}}
    """
    rec_prompt = ChatPromptTemplate.from_template(rec_prompt_template)
    return rec_prompt | llm | JsonOutputParser()

def process_user_query(user_query, llm, vector_db, recommendation_function, recommendation_models, session_state):
    """Routes the user's query to the correct handler."""
    router = get_router_chain(llm)
    intent = router.invoke({"query": user_query})

    if "follow_up_recommendation" in intent.lower():
        if session_state.get("last_recommendation") and session_state.get("chat_profile"):
            last_recommendation = session_state["last_recommendation"]
            user_profile = session_state["chat_profile"]
            
            explanation = generate_explanation_with_gemini(user_profile, last_recommendation, "English")
            return explanation, "follow_up_recommendation"
        else:
            # New logic to handle no recent recommendation
            return "Sure, what product would you like me to clarify for you?", "follow_up_recommendation"

    elif "recommendation" in intent.lower():
        known_jobs = list(recommendation_models["le_job"].classes_)
        known_regions = list(recommendation_models["le_region"].classes_)
        data_extraction_chain = get_data_extraction_chain(llm, known_jobs, known_regions)
        
        try:
            parsed_profile = data_extraction_chain.invoke({"input": user_query})
            
            # --- ✨ New Logic to check for all fields ---
            required_keys = ["Job", "Region", "Monthly_Income", "Number_of_Dependents"]
            missing_keys = [key for key in required_keys if not parsed_profile.get(key)]
            
            if missing_keys:
                # If any of the required fields are missing, prompt the user for them
                missing_str = ", ".join(missing_keys).replace("_", " ")
                return f"I need a little more information to give you a recommendation. Can you provide your {missing_str}?", "recommendation"
            
            # --- ✨ End of New Logic ---

            monthly_income = parsed_profile.get('Monthly_Income')
            if monthly_income is not None:
                monthly_income = float(monthly_income)
            
            user_profile = {
                'Job': parsed_profile.get('Job'),
                'Region': parsed_profile.get('Region'),
                'Monthly_Income': monthly_income,
                'Number_of_Dependents': parsed_profile.get('Number_of_Dependents'),
                'Max_Affordable_Premium': monthly_income * 0.1,
                'Claim_Frequency': 0
            }

            recs = recommendation_function(
                user_profile,
                recommendation_models["product_df"],
                recommendation_models["kmeans_model"],
                recommendation_models["scaler"],
                recommendation_models["le_job"],
                recommendation_models["le_region"],
                recommendation_models["cluster_product_map"]
            )

            if recs:
                session_state["chat_profile"] = user_profile
                session_state["last_recommendation"] = recs[0]

                rec_str = "Based on your information, here are some recommended plans:\n\n"
                for rec in recs:
                    rec_str += f"**- {rec['Product_Name']}**\n"
                    rec_str += f"  - **Monthly Premium:** ₦{rec['Monthly_Premium']:,.2f}\n"
                    rec_str += f"  - **Score:** {rec['Score']}%\n"
                    rec_str += f"  - **Reasoning:** {', '.join(rec['Reasons'])}\n\n"
                return rec_str, "recommendation"
            else:
                return "I'm sorry, I couldn't find a suitable recommendation for you based on the information provided.", "recommendation"
            
        except ValueError:
            return "There was an error processing your income. Please make sure it's a number (e.g., 40000).", "recommendation"
        except Exception as e:
            return f"There was an error processing your recommendation request: {e}", "recommendation"

    elif "general_question" in intent.lower():
        rag_chain = get_rag_chain(llm, vector_db)
        response = rag_chain.invoke({"input": user_query})
        return response['answer'], "general_question"

    return "I'm sorry, I couldn't understand your request. Can you please rephrase it?", "unknown"