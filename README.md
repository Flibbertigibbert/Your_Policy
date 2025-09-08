# YourPolicy - PolicyPal
This looks like a fantastic project\! Here's a complete, structured Markdown document for your GitHub project, incorporating all the information you've provided. I've organized it logically with headings, code blocks for clarity, and kept the language engaging and straightforward.

-----

# PolicyPal - AI-Powered Insurance for Underserved Populations

## Project Overview

PolicyPal is a groundbreaking, **multilingual, AI-powered insurance recommendation system**. It's specifically designed to empower **underserved populations**—such as artisans, gig workers, and microbusinesses—to make confident decisions about crucial insurance products. By seamlessly blending **clustering-based personalization**, **Gemini-powered explanations and translations**, and an intuitive **conversational assistant interface** built with Streamlit, PolicyPal bridges the gap in financial literacy and access to essential protection.

-----

## System Architecture

PolicyPal's architecture is designed for modularity and scalability, integrating advanced AI capabilities with a user-friendly interface.

### Components:

  * **Frontend:** A dynamic interface built with **Streamlit**, featuring user-friendly form inputs and an interactive sidebar chat assistant.
  * **Backend:** Robust **Python modules** handle the core recommendation logic, seamless translation services, intelligent routing, and efficient data management.
  * **AI Integration:** Leverages the power of **Gemini LLM and embeddings** for generating clear explanations, performing accurate translations, and classifying user intent.
  * **Data Layer:** Utilizes an **Excel-based product catalog** and pre-trained **serialized models** for clustering and encoding.

### Operational Flow:

1.  **User Input:** Users provide their profile information either through a structured form or via the chat assistant.
2.  **Intent Classification:** The system intelligently classifies the user's intent and extracts structured data from their input.
3.  **Recommendation Engine:** A sophisticated engine scores and filters relevant insurance products based on user profile and preferences.
4.  **AI-Generated Insights:** Gemini generates personalized explanations for recommended products and translates the user interface and assistant responses.
5.  **Conversational Response:** The chat assistant provides contextual responses, utilizing Retrieval-Augmented Generation (RAG) or direct Gemini interactions.

-----

## Module Descriptions

This section details the function and purpose of each core Python module within the PolicyPal project.

  * `app.py`:

      * The **main entry point** for the Streamlit application.
      * Handles the **initialization of session state** and the loading of essential models and data.
      * Manages the **rendering of the user interface**, the chat assistant's functionality, and the recommendation form.
      * **Integrates Gemini** for generating chat responses and performing dynamic translations.

  * `router/router_agent.py`:

      * Responsible for **routing user queries** based on their classified intent.
      * Utilizes **Gemini to classify intent** and extract structured data from user messages.
      * Implements **Retrieval-Augmented Generation (RAG)** to answer general insurance-related questions.
      * Manages **follow-up logic** and ensures **multilingual responses**.

  * `gemini_utils.py`:

      * **Initializes the Gemini model** using the provided API key.
      * Generates **clear, personalized explanations** for recommended insurance products.
      * Performs **UI and assistant response translations**, incorporating cultural nuances and domain-specific terminology.
      * **Parses structured JSON output** from Gemini responses for seamless integration.

  * `recommendation.py`:

      * Contains the **core recommendation logic**, employing a combination of clustering and rule-based scoring.
      * Features **fuzzy matching** to accurately map user job titles to known categories.
      * **Predicts user clusters** and filters products based on relevance and affordability.
      * Returns **scored recommendations** along with clear, interpretable reasons.

  * `utils.py`:

      * Provides **utility functions** for loading models and product data efficiently.
      * Leverages **Streamlit caching** for significant performance optimization.

  * `config.py`:

      * Acts as a **central configuration file**, defining key constants.
      * Includes settings such as the **affordability percentage threshold**, minimum recommendation score, and a list of **supported languages**.

-----

## Data & Models

PolicyPal relies on a carefully curated set of data files and pre-trained models to deliver personalized recommendations.

### Data Files:

  * `data/product_catalog.xlsx`: Contains comprehensive details of insurance products, including ID, name, premium, category, and other relevant attributes.
  * `data/job_keywords.json`: A mapping of job titles to relevant insurance keywords, crucial for filtering and scoring product relevance.
  * `data/insurance_policy.pdf`: This document serves as the knowledge base for Retrieval-Augmented Generation (RAG), enabling the system to answer general insurance questions.

### Model Files:

  * `models/le_job.pkl`: A **LabelEncoder** used for encoding categorical job titles.
  * `models/le_region.pkl`: A **LabelEncoder** for encoding categorical region information.
  * `models/scaler.pkl`: A **StandardScaler** instance used for scaling numerical features like income and number of dependents.
  * `models/kmeans_model.pkl`: The trained **KMeans clustering model** that groups users into segments based on their profile and interaction patterns.
  * `models/cluster_product_map.pkl`: A mapping that associates each cluster with product preference scores, guiding recommendation prioritization.

-----

## Language Support & Translation Features

PolicyPal is designed with inclusivity at its core, supporting multiple languages and offering sophisticated translation capabilities.

### Supported Languages:

PolicyPal proudly supports **five key languages**:

  * English
  * Nigerian Pidgin
  * Yoruba
  * Hausa
  * Igbo

### Advanced Translation Features:

  * **Powered by Gemini LLM:** Translations are generated dynamically using the advanced capabilities of the Gemini Large Language Model.
  * **Cultural Adaptation:** The system goes beyond literal translation to ensure **cultural appropriateness**, tone matching, and the use of domain-specific terminology for insurance, healthcare, and finance.
  * **Caching Mechanism:** An intelligent **caching mechanism** is employed to avoid redundant translations, significantly reducing latency and costs.
  * **Language-Specific Guidelines:** Dedicated guidelines are used to ensure that communication is always **respectful, relatable, and effective** across all supported languages.

-----

## Gemini Integration

Gemini is a pivotal component of PolicyPal, enhancing user experience and intelligence across several key areas:

1.  **Personalized Explanations:** Gemini provides **plain-language, personalized reasons** for why specific products are recommended to a user.
2.  **Dynamic Translations:** The **user interface and assistant responses** are translated with a keen awareness of cultural context and domain-specific language.
3.  **Natural Language Understanding (NLU) Parsing:** Gemini expertly **extracts structured data and identifies user intent** from conversational messages.
4.  **RAG for Knowledge Retrieval:** It powers the system's ability to **answer general insurance questions** by retrieving relevant information from the `insurance_policy.pdf` document.

### Gemini Model Used:

  * `gemini-2.5-flash-lite`

### Fallback Mechanism:

  * In the event of Gemini unavailability, the system gracefully **defaults to basic logic and English-only responses**, ensuring core functionality remains accessible.

-----

## Session State Usage

Streamlit's `session_state` is crucial for maintaining continuity and personalization throughout user interactions. Key variables stored include:

  * `chat_profile`: Stores the user's complete profile information (job, region, income, dependents).
  * `last_recommendation`: Holds the details of the most recently recommended product.
  * `conversation_history`: Maintains a log of chat messages for display and context.
  * `language`: Tracks the user's currently selected language for translation purposes.
  * `last_intent`: Records the last classified user intent to inform subsequent actions.
  * `last_missing_key`: Identifies any missing profile fields needed for follow-up prompts.
  * `i18n_cache`: Stores cached translations to optimize performance and reduce API calls.

-----

## Recommendation Logic

PolicyPal's recommendation logic is a sophisticated blend of user profiling, data analysis, and machine learning to deliver highly personalized insurance suggestions.

1.  **Data Loading and Preparation:** The process begins by loading data from multiple Excel files: `product_catalog.xlsx`, user behavioral data, user profiles, and claims data. These datasets are then merged and meticulously preprocessed. The `Browsed_Products` column, initially a string representation of a list, is converted into an actual list using `ast.literal_eval`. Missing values in `Rejected_Product` are handled by filling them with `0`, and numerical columns are converted to appropriate data types.

2.  **User Risk Profiling:** User profile and claims data are merged based on 'Job' and 'Region' to construct a `user_risk_profile` DataFrame. This DataFrame captures average claim amounts and claim frequencies for various job and region combinations. An **affordability threshold** (`Max_Affordable_Premium`) is calculated for each user, set at 10% of their monthly income.

3.  **Creating Interaction Matrix:** An `interaction_matrix` is generated to represent user-product interactions. This matrix indicates whether users have browsed a product (marked as `1`), rejected it (marked as `-1`), or had no interaction (marked as `0`).

4.  **Feature Engineering and Scaling:** The `user_risk_profile` and `interaction_matrix` are combined into a comprehensive `full_df`. Categorical features like 'Job' and 'Region' are encoded using `LabelEncoder`, while numerical features such as 'Monthly\_Income' and 'Number\_of\_Dependents' are scaled using `StandardScaler`.

5.  **Clustering:** **KMeans clustering** is applied to the scaled features (including encoded job and region, scaled numerical features, and product interactions). This process groups users into distinct clusters based on their profile and interaction patterns, allowing for segment-specific insights.

6.  **Fuzzy Matching for Job Titles:** To accommodate variations, typos, and alternative spellings in user-provided job titles, a **fuzzy matching function (`find_closest_job`)** is employed. This function compares the user's input against a predefined list of known job titles using the `fuzzywuzzy` library, identifying the closest match above a specified similarity threshold.

7.  **Generating Recommendations:** The `recommend_products_for_user` function takes a new user's profile as input:

      * It first uses the **fuzzy matching function** to find the closest known job. If no close match is identified, it provides generic recommendations (e.g., 'Health Micro Plan', 'Life Starter Plan').
      * If a close job match is found, it encodes the user's job and region (handling cases where the region might not have been present in the original training data).
      * It scales the user's numerical features.
      * It **predicts the cluster** the user belongs to based on their encoded and scaled features.
      * It identifies the **top products preferred by users within that cluster**, referencing the `cluster_product_map`.
      * Each potential product is then scored based on **affordability**, whether its premium fits within the user's budget, and its **relevance to the user's (fuzzy-matched) job** based on predefined keywords.
      * Recommendations are generated for products that are both relevant to the user's job and rank among the top preferred products in their identified cluster.

Essentially, PolicyPal leverages clustering to understand user segments and their product preferences, incorporates user-specific risk and affordability data, and uses fuzzy matching to enhance robustness against variations in user input, thereby delivering truly tailored recommendations.

-----

## Translation Logic

The translation process within PolicyPal is managed by the `translate_ui_with_gemini` function, ensuring accurate, culturally relevant, and context-aware communication:

  * **Core Functionality:** Uses Gemini to translate both UI elements and assistant responses.
  * **Language-Specific Adaptations:** Applies tailored language guidelines, including appropriate formality levels, cultural notes, and common linguistic patterns.
  * **Domain-Specific Guidance:** Incorporates specific terminology and context relevant to insurance, healthcare, and finance.
  * **Key Translation Goals:**
      * **Accuracy and Meaning Preservation:** Ensuring the original intent and meaning are faithfully conveyed.
      * **Cultural Adaptation:** Making the language resonate with the target audience's cultural background.
      * **Tone and Style Matching:** Maintaining a consistent and appropriate tone throughout the application.
      * **Localization:** Adapting elements like currency and date formats where necessary.
  * **Caching:** Leverages `session_state.i18n_cache` to store previously translated phrases, significantly improving performance and reducing costs.

-----

## Setup Instructions

Follow these steps to set up and run the PolicyPal application locally.

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd yourpolicy
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Create `.env` File:**
    Create a file named `.env` in the root directory of the project and add your Gemini API key:

    ```
    GEMINI_API_KEY=your_google_gemini_api_key_here
    ```

4.  **Prepare Data and Models:**
    Place the following files in their respective directories:

      * `data/product_catalog.xlsx`
      * `data/job_keywords.json`
      * `data/insurance_policy.pdf`
      * `models/le_job.pkl`
      * `models/le_region.pkl`
      * `models/scaler.pkl`
      * `models/kmeans_model.pkl`
      * `models/cluster_product_map.pkl`

5.  **Run the Application:**
    Start the Streamlit application from the `yourpolicy` directory:

    ```bash
    streamlit run app.py
    ```

-----

## Folder Structure

A clear understanding of the project's organization is key to efficient development and maintenance.

```
yourpolicy/
├── app.py               # Main Streamlit application entry point
├── config.py            # Configuration settings and constants
├── utils.py             # Utility functions (model loading, caching)
├── recommendation.py    # Core recommendation engine logic
├── gemini_utils.py      # Utilities for Gemini LLM integration
├── router/              # Module for query routing and NLU
│   └── router_agent.py  # Handles intent classification and RAG
├── data/                # Contains raw data files
│   ├── product_catalog.xlsx
│   ├── job_keywords.json
│   └── insurance_policy.pdf
├── models/              # Stores pre-trained machine learning models
│   ├── le_job.pkl
│   ├── le_region.pkl
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   └── cluster_product_map.pkl
├── .env                 # Environment variables (e.g., API keys)
└── requirements.txt     # Project dependencies
```

-----

## Contribution Guidelines

We welcome contributions to PolicyPal\! Please follow these guidelines to ensure a smooth and collaborative development process:

  * **Fork the Repository:** Create your feature branch (`git checkout -b feature/AmazingFeature`).
  * **Follow Modular Design:** Keep code organized and adhere to the existing module structure.
  * **Cultural Appropriateness:** Ensure all translations and explanations are culturally sensitive and accurate.
  * **Testing:** Test your changes with diverse user profiles and across multiple languages.
  * **Pull Request:** Submit a clear and concise pull request with a detailed description of your changes.

-----

## Future Improvements

We envision PolicyPal evolving to offer even more value. Potential areas for future development include:

  * **Real-time Interactions:** Integrate real-time claim tracking and agent chat functionalities.
  * **Dynamic Product Catalog:** Enable dynamic updates and expansion of the insurance product catalog.
  * **Analytics Dashboard:** Develop an analytics dashboard to monitor user engagement and recommendation performance.

-----

## User Experience (UX) Design Principles

PolicyPal's user experience is crafted with inclusivity, clarity, and ease of use in mind:

  * **Inclusive Language:** Employ simple, jargon-free explanations to ensure accessibility for all users.
  * **Multilingual UI:** Provide a language selector for a fully translated interface, making the system approachable for diverse linguistic backgrounds.
  * **Conversational Chat Assistant:** A dedicated sidebar assistant offers a natural, conversational way for users to interact with the system.
  * **Form-Based Input:** Offer structured forms for users who prefer direct data entry.
  * **Visual Feedback:** Utilize progress bars and visual highlights to provide clear feedback on recommendation scores and system status.
  * **Lead Generation:** Incorporate embedded contact forms for efficient lead generation and follow-up with insurance agents.

-----
