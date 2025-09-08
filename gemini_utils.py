import os
import json
import re
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from static_translations import STATIC_TRANSLATIONS

# Gemini import (only if available)
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

# # Static translation function
# def t(text: str, lang: str) -> str:
#     """Translate UI or short text snippets using static dictionary."""
#     if lang == "Pidgin":
#         return STATIC_TRANSLATIONS.get(text, text)
#     return text

# Gemini-based translation fallback (not used for static UI)
def translate_ui_with_gemini(text: str, target_language: str, context=None, user_profile=None, conversation_tone=None, domain="insurance") -> str:
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
    
        # Language-specific guidelines
    language_guidelines = {
        "Yoruba": {
            "formality": "Use appropriate honorifics and respectful language",
            "cultural_notes": "Include cultural greetings and family-centered language",
            "common_patterns": "Tone marks important, use familiar metaphors from Yoruba culture"
        },
        "Hausa": {
            "formality": "Use Islamic greetings and respectful address forms",
            "cultural_notes": "Consider Northern Nigerian cultural context",
            "common_patterns": "Arabic loanwords for formal concepts, respectful indirect communication"
        },
        "Igbo": {
            "formality": "Use appropriate respectful forms and community-oriented language",
            "cultural_notes": "Include cultural values of family and community",
            "common_patterns": "Use familiar proverbs and culturally relevant examples"
        },
        "Nigerian Pidgin": {
            "formality": "Balance respect with accessibility",
            "cultural_notes": "Mix with English appropriately, use local references",
            "common_patterns": "Simple structure, familiar expressions, cultural markers"
        }
    }
    
    # Domain-specific terminology guidance
    domain_guidance = {
        "insurance": {
            "key_concepts": "premium, coverage, benefits, claims, policy, deductible, beneficiary",
            "tone": "professional yet accessible, trustworthy, reassuring",
            "common_concerns": "cost, reliability, family protection, claims process"
        },
        "healthcare": {
            "key_concepts": "symptoms, treatment, medication, diagnosis, prevention",
            "tone": "caring, professional, clear, non-alarming",
            "common_concerns": "health outcomes, treatment costs, side effects"
        },
        "finance": {
            "key_concepts": "interest, investment, savings, loans, credit, budget",
            "tone": "professional, trustworthy, educational",
            "common_concerns": "financial security, returns, risks, fees"
        },
        "education": {
            "key_concepts": "curriculum, assessment, learning, skills, development",
            "tone": "encouraging, supportive, clear, motivational",
            "common_concerns": "progress, understanding, future opportunities"
        },
        "general": {
            "key_concepts": "varies by context",
            "tone": "adapt to conversation context",
            "common_concerns": "clarity, relevance, usefulness"
        }
    }
    
    # Get language-specific and domain-specific guidance
    lang_guide = language_guidelines.get(target_language, {
        "formality": "Use appropriate formal register",
        "cultural_notes": "Consider cultural context and local customs",
        "common_patterns": "Follow standard grammatical patterns and conventions"
    })
    
    domain_guide = domain_guidance.get(domain, domain_guidance["general"])

    prompt = f"""
    You are a professional translator specializing in {domain} communications with deep cultural and linguistic expertise. Your goal is to provide accurate, culturally appropriate translations that resonate with the target audience.

    ## TARGET LANGUAGE: {target_language}

    ## TRANSLATION CONTEXT
    Original Text: {text}
    Domain: {domain}
    Context: {context or 'General communication'}
    User Profile: {user_profile or 'General user'}
    Desired Tone: {conversation_tone or 'Professional and friendly'}

    LANGUAGE-SPECIFIC GUIDELINES

    Formality Level:
    {lang_guide['formality']}

    Cultural Considerations:
    {lang_guide['cultural_notes']}

    Language Patterns:
    {lang_guide['common_patterns']}

    DOMAIN EXPERTISE ({domain.upper()})
    Key Concepts to Handle:
    {domain_guide['key_concepts']}

    Appropriate Tone:
    {domain_guide['tone']}

    Common User Concerns:
    {domain_guide['common_concerns']}

    COMPREHENSIVE TRANSLATION PRINCIPLES

    1. Accuracy and Meaning Preservation
    - Maintain the exact meaning and intent of the original text
    - Preserve technical accuracy while making content accessible
    - Ensure no critical information is lost or altered

    2. Cultural Adaptation
    - Adapt cultural references to be relevant for {target_language} speakers
    - Use culturally appropriate examples and metaphors
    - Consider local customs, values, and communication styles
    - Adjust formality levels based on cultural norms

    3. Target Audience Awareness
    - Consider the education level and background of typical users
    - Use vocabulary appropriate for the intended audience
    - Balance technical accuracy with comprehensibility
    - Include explanations for complex concepts when needed

    4. Tone and Style Matching
    - Match the emotional tone of the original message
    - Maintain consistency with the specified conversation tone
    - Ensure the style is appropriate for the domain and context
    - Use language that builds trust and rapport

    5. Technical Term Handling
    - Translate technical terms accurately with explanations if needed
    - Use established terminology in the target language where available
    - Provide context for terms that don't have direct equivalents
    - Maintain consistency in terminology throughout

    6. Grammar and Structure
    - Follow proper grammar rules for {target_language}
    - Use sentence structures that flow naturally in the target language
    - Adapt punctuation and formatting conventions as needed
    - Ensure readability and clarity

    7. Localization Elements
    - Use appropriate currency, date, and number formats
    - Include relevant local references when helpful
    - Consider regional variations within the language
    - Adapt examples to local context

    QUALITY ASSURANCE CHECKLIST

    Before providing the translation, verify:
    - [ ] Meaning is accurately preserved
    - [ ] Cultural context is appropriate
    - [ ] Technical terms are handled correctly
    - [ ] Tone matches the intended communication style
    - [ ] Grammar and structure are correct
    - [ ] Text flows naturally in {target_language}
    - [ ] Target audience would easily understand the message
    - [ ] No critical information is missing or added

    SPECIAL INSTRUCTIONS FOR THIS TRANSLATION

    Context-Specific Adaptations:
    """ + (f"""
    **User Profile Considerations:**
    - Adapt language complexity based on user background: {user_profile}
    - Consider user's likely concerns and priorities
    - Use examples and references relevant to their situation
    """ if user_profile else "") + f"""

    Conversation Context:
    - This message appears in the context of: {context or 'general communication'}
    - Previous conversation tone and style should be maintained
    - Consider this message's role in the broader conversation flow

    Tone Specifications:
    - Primary tone: {conversation_tone or 'professional and friendly'}
    - Ensure emotional appropriateness for the content and context
    - Balance professionalism with approachability

    OUTPUT REQUIREMENTS

    Provide ONLY the translated text without additional commentary, explanations, or formatting. The translation should be concise and ready for immediate use in the {domain} application.

    """
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
