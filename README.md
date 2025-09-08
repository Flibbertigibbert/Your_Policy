# YourPolicy - PolicyPal

Project Overview

PolicyPal is a multilingual, AI-powered insurance recommendation system designed to help underserved populations such as artisans, gig workers, and microbusinesses make informed decisions about insurance products. It combines clustering-based personalization, Gemini-powered explanations and translations, and a conversational assistant interface built with Streamlit.

## System Architecture

Components:
•	Frontend: Built with Streamlit, includes form inputs and a sidebar chat assistant.
•	Backend: Python modules for recommendation logic, translation, routing, and data handling.
•	AI Integration: Gemini LLM and embeddings for explanations, translations, and intent classification.
•	Data Layer: Excel-based product catalog and serialized models for clustering and encoding.
Flow:
1.	User provides profile via form or chat.
2.	System classifies intent and extracts structured data.
3.	Recommendation engine scores and filters products.
4.	Gemini generates explanations and translates UI.
5.	Chat assistant responds contextually using RAG or Gemini.

## Module Descriptions
 
app.py
•	Main entry point for the Streamlit app.
•	Initializes session state and loads models/data.
•	Handles UI rendering, chat assistant, and recommendation form.
•	Integrates Gemini for chat responses and translations.
router_agent.py
•	Routes user queries based on intent.
•	Uses Gemini to classify intent and extract structured data.
•	Implements RAG for general insurance questions.
•	Handles follow-up logic and multilingual responses.
gemini_utils.py
•	Initializes Gemini model using API key.
•	Generates explanations for recommended products.
•	Translates UI and assistant responses with cultural and domain-specific adaptation.
•	Parses structured JSON from Gemini responses.
recommendation.py
•	Core recommendation logic using clustering and rule-based scoring.
•	Fuzzy matches user job to known jobs.
•	Predicts cluster and filters products by relevance and affordability.
•	Returns scored recommendations with reasons.
utils.py
•	Utility functions for loading models and product data.
•	Uses Streamlit caching for performance optimization.
config.py
•	Central configuration file.
•	Defines constants like affordability percentage, minimum score, and supported languages.


## Data & Models

 Data Files
•	product_catalog.xlsx: Contains insurance product details including ID, name, premium, and category.
•	job_keywords.json: Maps job titles to relevant insurance keywords for filtering and scoring.
•	insurance_policy.pdf: Used for RAG-based responses to general insurance questions.
Model Files
•	le_job.pkl: LabelEncoder for job titles.
•	le_region.pkl: LabelEncoder for regions.
•	scaler.pkl: StandardScaler for income and dependents.
•	kmeans_model.pkl: KMeans clustering model trained on user-product features.
•	cluster_product_map.pkl: Maps clusters to product preference scores.
Language Support
PolicyPal supports five languages:
•	English
•	Nigerian Pidgin
•	Yoruba
•	Hausa
•	Igbo
Translation Features:
•	Powered by Gemini LLM.
•	Includes cultural adaptation, tone matching, and domain-specific terminology.
•	Uses a caching mechanism to avoid redundant translations.
•	Language-specific guidelines ensure respectful and relatable communication.

## Gemini Integration

Gemini is used for:
1.	Explanations: Personalized, plain-language reasons for product recommendations.
2.	Translations: UI and assistant responses are translated with cultural and domain awareness.
3.	NLU Parsing: Extracts structured data and intent from user messages.
4.	RAG Responses: Answers general insurance questions using context from insurance_policy.pdf.
Gemini Model Used:
•	gemini-2.5-flash-lite
Fallbacks:
•	If Gemini is unavailable, the system defaults to basic logic and English responses.
Session State Usage
Streamlit’s session_state is used to maintain continuity and personalization across interactions.
Keys:
•	chat_profile: Stores user profile (job, region, income, dependents).
•	last_recommendation: Stores the most recent recommended product.
•	conversation_history: Maintains chat messages for display.
•	language: Tracks selected language for translation.
•	last_intent: Tracks last classified user intent.
•	last_missing_key: Tracks missing profile fields for follow-up prompts.
•	i18n_cache: Caches translations to reduce latency and cost.


## Recommendation Logic

The recommendation logic in this project works by combining user profile information with product data and claims data to create personalized recommendations. Here's a breakdown of the steps:

Data Loading and Preparation: The process starts by loading data from four different Excel files: product catalog, user behavioral data, user profiles, and claims data. These datasets are then merged and preprocessed. The 'Browsed_Products' column in the user behavioral data is converted from a string representation of a list to an actual list using ast.literal_eval. Missing values in 'Rejected_Product' are filled with 0, and numerical columns are converted to appropriate data types.

User Risk Profiling: The user profile and claims data are merged based on 'Job' and 'Region' to create a user_risk_profile DataFrame. This DataFrame includes information about average claim amounts and claim frequencies for different job and region combinations. An affordability threshold (Max_Affordable_Premium) is calculated for each user based on 10% of their monthly income.

Creating Interaction Matrix: An interaction_matrix is created to represent user-product interactions. This matrix shows which products users have browsed (marked with 1), which they have rejected (marked with -1), and which they haven't interacted with (marked with 0).

Feature Engineering and Scaling: The user_risk_profile and interaction_matrix are combined into a full_df. Categorical features ('Job' and 'Region') are encoded using LabelEncoder, and numerical features ('Monthly_Income', 'Number_of_Dependents', 'Max_Affordable_Premium') are scaled using StandardScaler.
Clustering: KMeans clustering is applied to the scaled features (including encoded job and region, scaled numerical features, and product interactions). This groups users into clusters based on their profile and interaction patterns.

Fuzzy Matching for Job Titles: To handle variations and typos in user-provided job titles, a fuzzy matching function (find_closest_job) is used. This function compares the user's job input to a predefined list of known job titles using fuzzywuzzy and finds the closest match above a specified similarity threshold.
Generating Recommendations: The recommend_products_for_user function takes a new user's profile as input.
It first uses the fuzzy matching function to find the closest known job. If no close match is found, it provides generic recommendations ('Health Micro Plan' and 'Life Starter Plan').

If a close job match is found, it encodes the job and region (handling cases 
where the region might not have been in the original training data).
It scales the user's numerical features.
It predicts the cluster the user belongs to based on their encoded and scaled features.
It identifies the top products preferred by users in that cluster using the cluster_product_map.
Each product is scored based on affordability, whether it fits within the user's premium budget, and its relevance to the user's (fuzzy-matched) job based on predefined keywords.
Recommendations are generated for products that are both relevant to the user's job and are among the top products in their cluster.

Essentially, the system leverages clustering to understand user segments and their product preferences, incorporates user-specific risk and affordability data, and uses fuzzy matching to make the system more robust to variations in user input, providing tailored recommendations.

## Translation Logic

Translation is handled by translate_ui_with_gemini, which:
-	Uses Gemini to translate UI and assistant responses.
-	Applies language-specific guidelines (formality, cultural notes, patterns).
-	Applies domain-specific guidance (insurance, healthcare, finance, etc.).
-	Ensures:
  --	Accuracy and meaning preservation
  --	Cultural adaptation
  --	Tone and style matching
  --	Localization (currency, date formats)
-	Caches translations using session_state.i18n_cache.

## Setup Instructions
 
1. Install dependencies
pip install -r requirements.txt

3. Create a .env file
GEMINI_API_KEY=your_google_gemini_api_key_here

5. Prepare your data and models
Place the following files in the appropriate folders:
•	data/product_catalog.xlsx
•	data/job_keywords.json
•	data/insurance_policy.pdf
•	models/le_job.pkl
•	models/le_region.pkl
•	models/scaler.pkl
•	models/kmeans_model.pkl
•	models/cluster_product_map.pkl

7. Run the app
streamlit run app.py

## Folder Structure
'''
yourpolicy/
├── app.py
├── config.py
├── utils.py
├── recommendation.py
├── gemini_utils.py
├── router/
│   └── router_agent.py
├── data/
│   ├── product_catalog.xlsx
│   ├── job_keywords.json
│   └── insurance_policy_catalog.pdf
├── models/
│   ├── le_job.pkl
│   ├── le_region.pkl
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   └── cluster_product_map.pkl
├── .env
└── requirements.txt
'''

Contribution Guidelines
•	Fork the repository and create a feature branch.
•	Follow modular design principles.
•	Ensure translations and explanations are culturally appropriate.
•	Test with diverse user profiles and languages.
•	Submit a pull request with a clear description.


Future Improvements
•	Integrate real-time claim tracking and agent chat.
•	Expand product catalog with dynamic updates.
•	Add analytics dashboard for user engagement and recommendation performance.
 User Experience Design
•	Inclusive Language: Simple, jargon-free explanations.
•	Multilingual UI: Language selector with full-page translation.
•	Chat Assistant: Sidebar assistant for conversational interaction.
•	Form-Based Input: Structured form for users who prefer direct entry.
•	Visual Feedback: Progress bars and highlights for recommendation scores.
•	Contact Forms: Embedded forms for lead generation and agent follow-up.



