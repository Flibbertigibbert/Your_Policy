import math

from sklearn.preprocessing import LabelEncoder

import pandas as pd

from fuzzywuzzy import fuzz

def find_closest_job(user_job, known_jobs):
    """
    Finds the closest job title in a list of known jobs using fuzzy matching.

        user_job: The job title provided by the user (string).
        known_jobs: A list of known job titles from the job_keywords dictionary (list of strings).

        The closest matching job title from known_jobs, or None if no good match is found.
    """
    best_match = None
    highest_score = -1

    # Define the similarity threshold
    similarity_threshold = 75 # This threshold can be adjusted based on desired strictness

    for known_job in known_jobs:
        score = fuzz.token_set_ratio(user_job.lower(), known_job.lower())
        if score > highest_score:
            highest_score = score
            best_match = known_job

    # Check if the highest score is below the threshold
    if highest_score < similarity_threshold:
        return None # No sufficiently close match found

    return best_match


def recommend_products_for_user(user_profile, product_df, kmeans_model, scaler, le_job, le_region, cluster_product_map, job_keywords):
    """
    Recommend products for a new user based on profile and product features.

    - user_profile: dict with keys 'Job', 'Monthly_Income', 'Max_Affordable_Premium', 'Number_of_Dependents', 'Region'
    - product_df: DataFrame with product features
    - kmeans_model: trained KMeans model from joint clustering
    - scaler: fitted StandardScaler for numerical features
    - le_job, le_region: fitted LabelEncoders for categorical features
    - cluster_product_map: DataFrame with product scores per cluster
    - job_keywords: Dictionary mapping job titles to relevant product keywords

    Returns:
    - List of recommended products with percentage scores and explanations
    """
    max_score = 4  

    # Find the closest job using fuzzy matching
    known_jobs = list(job_keywords.keys())
    fuzzy_matched_job = find_closest_job(user_profile['Job'], known_jobs)

    # Handle case where no close job match is found - provide generic recommendations
    if fuzzy_matched_job is None:
        print(f"No close job match found for '{user_profile['Job']}' above the threshold. Providing generic recommendations.")
        # Return generic recommendations: Health Micro Plan and Life Starter Plan
        generic_recommendations = []
        for _, product in product_df.iterrows():
            if product['Product_ID'] in ['P003', 'P005']:
                 # Assign a default score and reasons for generic recommendations
                generic_recommendations.append({
                    'Product_ID': product['Product_ID'],
                    'Product_Name': product['Product_Name'],
                    'Score': 50, # Default score for generic recommendations
                    'Monthly_Premium': product['Monthly_Premium'],
                    'Reasons': ["Your Everyday Health needs", "Basic Life coverage"]
                })
        return generic_recommendations


    current_job = fuzzy_matched_job

    # Create manual mappings for job and region based on the fitted encoders' classes
    job_mapping = {cls: idx for idx, cls in enumerate(le_job.classes_)}
    region_mapping = {cls: idx for idx, cls in enumerate(le_region.classes_)}

    # Get encoded job and region, using -1 for unseen values in the original training data
    job_encoded = job_mapping.get(current_job, -1)
    region_encoded = region_mapping.get(user_profile['Region'], -1)



    # Scale numerical features
    # Create a DataFrame with feature names for scaling
    numerical_data = pd.DataFrame([[user_profile['Monthly_Income'],
                                    user_profile['Number_of_Dependents'],
                                    user_profile['Max_Affordable_Premium']]],
                                  columns=['Monthly_Income', 'Number_of_Dependents', 'Max_Affordable_Premium'])
    scaled_values = scaler.transform(numerical_data)[0]

    # Create full feature vector as a DataFrame with feature names
    # Include region_encoded even if -1, as the model was trained with encoded regions
    feature_cols = [
        'Job_Encoded', 'Region_Encoded',
        'Monthly_Income', 'Number_of_Dependents',
         'Max_Affordable_Premium',
        'P001', 'P002', 'P003', 'P004', 'P005'
    ]
    user_data_df = pd.DataFrame([[job_encoded, region_encoded] + list(scaled_values) + [0, 0, 0, 0, 0]],
                                columns=feature_cols)


    # Predict cluster
    cluster_id = kmeans_model.predict(user_data_df)[0]

    # Get top products from cluster
    top_products = cluster_product_map.loc[cluster_id].sort_values(ascending=False).index.tolist()

    # Job relevance mapping
    relevant_keywords = job_keywords.get(current_job, [])

    # Score and explain each product
    recommendations = []
    for _, product in product_df.iterrows():
        # Filter out products not relevant to the user's job
        if not any(keyword.lower() in product['Product_Name'].lower() for keyword in relevant_keywords):
            continue 

        score = 0
        reasons = []

        # Affordability
        if product['Monthly_Premium'] <= user_profile['Monthly_Income'] * 0.1:
            score += 1
            reasons.append("Affordable based on your income")

        if product['Monthly_Premium'] <= user_profile['Max_Affordable_Premium']:
            score += 1
            reasons.append("Fits within your premium budget")

        # Job relevance (already passed filter)
        score += 1
        reasons.append(f"Relevant to your job as a {current_job}")


        # Only include products from top cluster preferences
        if product['Product_ID'] in top_products:
            recommendations.append({
                'Product_ID': product['Product_ID'],
                'Product_Name': product['Product_Name'],
                'Score': int(round((score / max_score) * 100, 0)),  # Percentage score with 0 decimals
                'Monthly_Premium': product['Monthly_Premium'],
                'Reasons': reasons
            })

    # Sort by score
    recommendations = sorted(recommendations, key=lambda x: x['Score'], reverse=True)

    # # Bundle logic: if exactly two products are recommended
    # if len(recommendations) == 2:
    #     premium_1 = recommendations[0]['Monthly_Premium']
    #     premium_2 = recommendations[1]['Monthly_Premium']
    #     bundle_premium = round(premium_1 * 1.0 + premium_2 * 0.2, 2)

    #     # Check affordability
    #     if bundle_premium <= user_profile['Max_Affordable_Premium']:
    #         bundle_name = f"{recommendations[0]['Product_Name']} + {recommendations[1]['Product_Name']} Bundle"
    #         bundle_score = int(round((recommendations[0]['Score'] + recommendations[1]['Score']) / 2, 0))
    #         bundle_reasons = list(set(recommendations[0]['Reasons'] + recommendations[1]['Reasons']))
    #         bundle_id = f"{recommendations[0]['Product_ID']}_{recommendations[1]['Product_ID']}"

    #         bundled_product = {
    #             'Product_ID': bundle_id,
    #             'Product_Name': bundle_name,
    #             'Score': bundle_score,
    #             'Monthly_Premium': bundle_premium,
    #             'Reasons': bundle_reasons
    #         }

    #         recommendations = [bundled_product]
    #     else:
    #         # Bundle not affordable—return only the first product
    #         recommendations = [recommendations[0]]


    return recommendations

def filter_recommendations(recs, cap, min_score=50):
    recs = recs or []
    affordable = [r for r in recs if float(r.get('Monthly_Premium', math.inf)) <= cap]
    scored = [r for r in affordable if float(r.get('Score', 0)) >= min_score]
    diag = {
        "total": len(recs),
        "affordable": len(affordable),
        "scored": len(scored)
    }
    return scored, diag
