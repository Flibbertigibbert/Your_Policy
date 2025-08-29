import os
import json
import re
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from static_translations import STATIC_TRANSLATIONS
from validators import missing_fields

# Optional Gemini import (only if available)
try:
    from google.generativeai import configure, GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Gemini API setup
def _get_gemini_key():
    k = os.getenv("GEMINI_API_KEY", "").strip()
    return k if k and len(k) > 20 else None

GEMINI_KEY = _get_gemini_key()
gemini_model = None
if GEMINI_AVAILABLE and GEMINI_KEY:
    try:
        configure(api_key=GEMINI_KEY)
        gemini_model = GenerativeModel("gemini-2.5-flash-lite")
    except Exception as e:
        st.sidebar.error(f"Gemini init failed: {e}")

# Static translation function
def t(text: str, lang: str) -> str:
    """Translate UI or short text snippets using static dictionary."""
    if lang == "Pidgin":
        return STATIC_TRANSLATIONS.get(text, text)
    return text

# Gemini-based translation fallback (not used for static UI)
def translate_ui_with_gemini(text: str, target_language: str) -> str:
    if target_language == "English" or not text.strip():
        return text
    if "i18n_cache" not in st.session_state:
        st.session_state.i18n_cache = {}
    cache_key = (text, target_language)
    if cache_key in st.session_state.i18n_cache:
        return st.session_state.i18n_cache[cache_key]
    if gemini_model is None:
        st.session_state.i18n_cache[cache_key] = text
        return text
    prompt = f"Translate to Nigerian Pidgin in a clear, friendly, natural tone. Keep it short. Text:\n\n{text}"
    try:
        resp = gemini_model.generate_content(prompt)
        out = (resp.text or text).strip()
    except Exception:
        out = text
    st.session_state.i18n_cache[cache_key] = out
    return out

# Explanation generator
def generate_explanation_with_gemini(user_profile, recommendation, target_language="English"):
    if gemini_model is None:
        base = (
            f"This plan ({recommendation.get('Product_Name')}) fits your budget and profile "
            f"as a {user_profile.get('Job')} in {user_profile.get('Region')} with "
            f"{user_profile.get('Number_of_Dependents')} dependents."
        )
        return base if target_language == "English" else translate_ui_with_gemini(base, target_language)

    prompt = f"""
Act as a friendly insurance guide for underserved users (artisans, gig workers, microbusinesses).
Use very plain language; avoid jargon. Be concise (2–4 sentences). Respond in {target_language}.

User Profile
- Job: {user_profile.get('Job')}
- Region: {user_profile.get('Region')}
- Monthly Income: ₦{user_profile.get('Monthly_Income')}
- Dependents: {user_profile.get('Number_of_Dependents')}
- Max Affordable Premium: ₦{user_profile.get('Max_Affordable_Premium'):.2f}

Recommended Product
- Name: {recommendation.get('Product_Name')}
- Monthly Premium: ₦{recommendation.get('Monthly_Premium')}
- Reasons: {', '.join(recommendation.get('Reasons', []))}

Provide a friendly explanation for why this product is suitable.
"""
    try:
        resp = gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Explanation unavailable: {e})"

# JSON parsing helper
def parse_json_safely(text: str):
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

def extract_profile_and_intent(user_message, known_profile, last_intent=None, last_missing_key=None, job_list=[], region_list=[]) -> dict:
    if gemini_model is None:
        # Fallback logic remains the same
        return {
            "intent": "recommendation" if "recommend" in user_message.lower() else "other",
            "fields": {},
            "plan_name": "",
            "product_type": "",
            "clarifying_question": ""
        }

    # The refined, multi-intent prompt starts here
    prompt = f"""
You are an insurance chatbot's Natural Language Understanding (NLU) engine. Your task is to accurately classify the user's intent and extract all relevant information in a strict JSON format.

Instructions:
1.  **Analyze the user's message** and the provided context.
2.  **Determine the intent**: Select the single best option from the following list: `recommendation`, `plan_info`, `claims_process`, `smalltalk`, `other`.
3.  **Extract relevant fields**:
    -   For `recommendation` intent, extract `job`, `region`, `income` (as a number), and `dependents` (as a number).
    -   For `plan_info`, extract the specific `plan_name`.
    -   For `claims_process`, extract the `product_type` of the claim.
4.  **Handle missing information**: If a field is not mentioned, use `null`. Do not make assumptions.
5.  **Generate a clarifying question**: If the determined intent is `recommendation` but key profile fields are missing, generate a concise `clarifying_question` to ask the user for the first missing piece of information.

Context:
- Known profile: {json.dumps(known_profile)}
- Last intent: {last_intent}
- Last missing key: {last_missing_key}
- User message: "{user_message}"

Examples:
(Start of examples)

Example 1: Recommendation with full profile
User message: "I'm a carpenter from Lagos with a ₦150,000 monthly income and 2 kids. What plan can you recommend?"
JSON response:
{{
  "intent": "recommendation",
  "fields": {{ "job": "carpenter", "region": "Lagos", "income": 150000.0, "dependents": 2 }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}

Example 2: Recommendation with missing info
User message: "I'm a driver and I have a family."
JSON response:
{{
  "intent": "recommendation",
  "fields": {{ "job": "driver", "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": "What region do you live in?"
}}

Example 3: Plan information request
User message: "Tell me about the Micro Health Plan."
JSON response:
{{
  "intent": "plan_info",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": "Micro Health Plan",
  "product_type": null,
  "clarifying_question": null
}}

Example 4: Claims process inquiry
User message: "How do I file a claim for my business insurance?"
JSON response:
{{
  "intent": "claims_process",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": "business insurance",
  "clarifying_question": null
}}

Example 5: Smalltalk / Greeting
User message: "Good morning! How are you?"
JSON response:
{{
  "intent": "smalltalk",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}

Example 6: Unclear/Other intent
User message: "I need to speak to someone."
JSON response:
{{
  "intent": "other",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}


Example 7: Smalltalk / Greeting
User message: "Good morning! How are you?"
JSON response:
{{
  "intent": "smalltalk",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}

Example 8: More Smalltalk examples
User message: "Hello there"
JSON response:
{{
  "intent": "smalltalk",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}

Example 9: Even more Smalltalk
User message: "hi"
JSON response:
{{
  "intent": "smalltalk",
  "fields": {{ "job": null, "region": null, "income": null, "dependents": null }},
  "plan_name": null,
  "product_type": null,
  "clarifying_question": null
}}

(End of examples)

Respond in STRICT JSON:
{{
  "intent": "...",
  "fields": {{ "job": "...", "region": "...", "income": ..., "dependents": ... }},
  "plan_name": "...",s
  "product_type": "...",
  "clarifying_question": "..."
}}
"""

    try:
        response = gemini_model.generate_content(prompt)
        data = parse_json_safely(response.text)

        fields = data.get("fields", {})
        # Normalize income and dependents
        if isinstance(fields.get("income"), str):
            fields["income"] = float(re.sub(r"[^\d.]", "", fields["income"]))
        if isinstance(fields.get("dependents"), str):
            fields["dependents"] = int(re.sub(r"[^\d]", "", fields["dependents"]))

        return {
            "intent": data.get("intent", "other"),
            "fields": {
                "job": fields.get("job", ""),
                "region": fields.get("region", ""),
                "income": fields.get("income", None),
                "dependents": fields.get("dependents", None)
            },
            "plan_name": data.get("plan_name", "").strip(),
            "product_type": data.get("product_type", "").strip(),
            "clarifying_question": data.get("clarifying_question", "")
        }

    except Exception:
        return {
            "intent": "other",
            "fields": {},
            "plan_name": "",
            "product_type": "",
            "clarifying_question": ""
        }