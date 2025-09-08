# app.py

import nest_asyncio
nest_asyncio.apply() 

__import__('pysqlite3')
import sys               # Import sys module
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import threading
import os
from dotenv import load_dotenv, find_dotenv
import time
import json
from config import AFFORDABILITY_PCT, MIN_SCORE, SUPPORTED_LANGS
from utils import load_joblib, load_products
from gemini_utils import  generate_explanation_with_gemini, translate_ui_with_gemini
from recommendation import  recommend_products_for_user, filter_recommendations
from router.router_agent import process_user_query, get_vector_db
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Global variables for session state keys
REC_KEY = "last_recommendation"
PROFILE_KEY = "chat_profile"

def init_session_state():
    """Initializes Streamlit session state variables."""
    defaults = {
        "last_intent": None,
        "last_missing_key": None,
        PROFILE_KEY: {},  # User profile for context
        REC_KEY: None,    # The last recommended product for context
        "chat_profile": {},
        "conversation_history": [],
        "user_input": "",
        "response_ready": False,
        "language": "English",
        "last_recommendation": None 
           }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_gemini_llm(api_key):
    """Initializes and caches the Gemini LLM by explicitly passing the API key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
        google_api_key=api_key
    )

def get_gemini_embeddings(api_key):
    """Initializes and caches the Gemini embeddings by explicitly passing the API key."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

init_session_state()

SESSION_TIMEOUT = 20

if "session_start_time" not in st.session_state:
    st.session_state["session_start_time"] = time.time()
elif time.time() - st.session_state["session_start_time"] > SESSION_TIMEOUT:
    st.session_state.clear()
    st.session_state["session_start_time"] = time.time()

# Load API Key securely
load_dotenv(find_dotenv(), override=True)

try:
    # Attempt to get API key from Streamlit secrets (for Streamlit Cloud deployment)
    google_api_key = st.secrets["GEMINI_API_KEY"]
except (AttributeError, KeyError):
    # Fallback to environment variable (for local testing or Colab)
    google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    st.error("Google API Key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

with open("data/job_keywords.json", "r") as f:
    job_keywords = json.load(f)

# Load models and data for the form.
try:
    le_job = load_joblib('models/le_job.pkl')
    le_region = load_joblib('models/le_region.pkl')
    scaler = load_joblib('models/scaler.pkl')
    kmeans_model = load_joblib('models/kmeans_model.pkl')
    cluster_product_map = load_joblib('models/cluster_product_map.pkl')
    product_df = load_products('data/product_catalog.xlsx')
except Exception as e:
    le_job = le_region = scaler = kmeans_model = cluster_product_map = product_df = None
    st.error(f"Model/Data load error: {e}")
    st.stop()

def t(text):
    return translate_ui_with_gemini(text, st.session_state.language)

# Streamlit UI Setup
st.set_page_config(page_title="YourPolicy Recommender", layout="centered")
st.title("YourPolicy – PolicyPal ")
st.markdown("Empowering underserved users to make informed insurance choices.")

st.sidebar.selectbox(
    "Language: /English/Pidgin/Yoruba/Hausa/Igbo",
    SUPPORTED_LANGS,
    index=SUPPORTED_LANGS.index(st.session_state.language),
    key="language"
)
language = st.session_state.language
st.sidebar.title(t("YourPolicy Chat Assistant"))

# Display chat history
for msg in st.session_state.conversation_history:
    with st.sidebar.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.sidebar.chat_input(t("Ask YourPolicy Assistant...")):
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.sidebar.chat_message("user"):
        st.markdown(user_input)

    with st.sidebar.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = get_gemini_llm(google_api_key)
            embeddings = get_gemini_embeddings(google_api_key)
            vector_db = get_vector_db(embeddings) # Pass the embeddings object, not the key

            response_tuple = process_user_query(
                user_query=user_input,
                llm=llm,
                vector_db=vector_db,
                recommendation_function=recommend_products_for_user,
                recommendation_models={
                    "product_df": product_df,
                    "kmeans_model": kmeans_model,
                    "scaler": scaler,
                    "le_job": le_job,
                    "le_region": le_region,
                    "cluster_product_map": cluster_product_map
                },
                session_state=st.session_state 
            )

            response = response_tuple[0]  
            st.markdown(response)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

st.divider()
st.subheader(t("Tell us about yourself"))

if (le_job is None) or (le_region is None) or (product_df is None):
    st.error(t("Models or product data not loaded. Please check your files."))
else:
    with st.form("user_input_form"):
        job = st.text_input(t("Enter your job: e.g., tailor, driver, trader"))
        region = st.text_input(t("Enter your Location: e.g Lagos, Abuja,Kano "))
        income = st.number_input(t("Monthly Income (₦)"), min_value=1000.0, step=10000.0)
        dependents = st.number_input(t("Number of Dependents"), min_value=0, step=1)
        submitted = st.form_submit_button(t("Get Recommendations"))

    if submitted:
        cap = income * (AFFORDABILITY_PCT / 100.0)
        st.write(t(f"Maximum Affordable Premium (10% of income): ₦{cap:,.2f}"))

        user_profile = {
            'Job': job,
            'Region': region,
            'Monthly_Income': income,
            'Number_of_Dependents': dependents,
            'Max_Affordable_Premium': cap
        }

        recs = recommend_products_for_user(
            user_profile, product_df, kmeans_model, scaler,
            le_job, le_region, cluster_product_map,job_keywords
        )

        to_show, diag = filter_recommendations(recs, cap, MIN_SCORE)

        st.subheader(t("Recommended Insurance Products"))
        for i, rec in enumerate(to_show):
            st.markdown(f"### {rec['Product_Name']}")
            st.markdown(t("**Monthly Premium:**") + f" ₦{float(rec['Monthly_Premium']):,.2f}")
            st.markdown(("**Score:**") + f" {float(rec['Score']):.0f}%")
            st.progress(min(max(float(rec['Score']) / 100.0, 0), 1))
            st.markdown(f"**{t('Why this is recommended:')}**")

            for reason in rec.get('Reasons', []):
                st.write(f"- {t(reason)}")

            explanation = generate_explanation_with_gemini(user_profile, rec, target_language=language)
            st.markdown(f"**{t('Explanation')}**")
            st.info({t(explanation)})

            # Contact form
            with st.expander(t("Interested? Click to provide your contact details")):
                with st.form(f"contact_form_{i}"):
                    name = st.text_input(t("Your Name"))
                    phone = st.text_input(t("Phone Number"))
                    preferred_time = st.selectbox(
                        t("Preferred Contact Time"),
                        ["Morning", "Afternoon", "Evening"]
                    )
                    submit_contact = st.form_submit_button(t("Submit"))

                    if submit_contact:
                        # You can store this info in session_state or send to a backend
                        contact_info = {
                            "Product": rec["Product_Name"],
                            "Name": name,
                            "Phone": phone,
                            "Preferred_Time": preferred_time
                        }
                        st.success(t("Thank you! An agent will contact you soon."))
        else:
            if diag["affordable"] == 0:
                warning_text = f"No plans fit your budget. Your affordability cap is ₦{cap:,.2f} (fixed at {AFFORDABILITY_PCT}% of income)."
                st.warning(t(warning_text))

