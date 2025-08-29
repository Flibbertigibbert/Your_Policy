import math

def recommend_products_for_user(user_profile, product_df, kmeans_model, scaler, le_job, le_region, cluster_product_map):
    max_score = 4
    job_encoded = le_job.transform([user_profile['Job']])[0]
    region_encoded = le_region.transform([user_profile['Region']])[0]
    scaled_values = scaler.transform([[user_profile['Monthly_Income'],
                                       user_profile['Number_of_Dependents'],
                                       user_profile['Max_Affordable_Premium']]])[0]
    user_vector = [job_encoded, region_encoded] + list(scaled_values) + [0, 0, 0, 0, 0]
    cluster_id = kmeans_model.predict([user_vector])[0]
    top_products = cluster_product_map.loc[cluster_id].sort_values(ascending=False).index.tolist()

    job_keywords = {
        'Tailor': ['Accident', 'Health'],
        'Driver': ['Accident', 'Health'],
        'Vendor': ['Burglary', 'Fire', 'Health'],
        'POS Agent': ['Device', 'Health'],
        'Mechanic': ['Accident', 'Device'],
        'Hairdresser': ['Health', 'Life'],
        'Electrician': ['Accident', 'Fire']
    }

    relevant_keywords = job_keywords.get(user_profile['Job'], [])
    recommendations = []
    for _, product in product_df.iterrows():
        if not any(keyword.lower() in product['Product_Name'].lower() for keyword in relevant_keywords):
            continue
        score = 0
        reasons = []
        if product['Monthly_Premium'] <= user_profile['Monthly_Income'] * 0.1:
            score += 1
            reasons.append("Affordable based on your income")
        if product['Monthly_Premium'] <= user_profile['Max_Affordable_Premium']:
            score += 1
            reasons.append("Fits within your premium budget")
        score += 1
        reasons.append(f"Relevant to your job as a {user_profile['Job']}")
        if product['Product_ID'] in top_products:
            recommendations.append({
                'Product_ID': product['Product_ID'],
                'Product_Name': product['Product_Name'],
                'Score': int(round((score / max_score) * 100, 0)),
                'Monthly_Premium': product['Monthly_Premium'],
                'Reasons': reasons
            })

    recommendations = sorted(recommendations, key=lambda x: x['Score'], reverse=True)
    if len(recommendations) == 2:
        premium_1 = recommendations[0]['Monthly_Premium']
        premium_2 = recommendations[1]['Monthly_Premium']
        bundle_premium = round(premium_1 * 1.0 + premium_2 * 0.2, 2)
        if bundle_premium <= user_profile['Max_Affordable_Premium']:
            bundle_name = f"{recommendations[0]['Product_Name']} (with added {recommendations[1]['Product_Name']} benefit)"
            bundle_score = int(round((recommendations[0]['Score'] + recommendations[1]['Score']) / 2, 0))
            bundle_reasons = list(set(recommendations[0]['Reasons'] + recommendations[1]['Reasons']))
            bundle_id = f"{recommendations[0]['Product_ID']}_{recommendations[1]['Product_ID']}"
            bundled_product = {
                'Product_ID': bundle_id,
                'Product_Name': bundle_name,
                'Score': bundle_score,
                'Monthly_Premium': bundle_premium,
                'Reasons': bundle_reasons
            }
            recommendations = [bundled_product]
        else:
            recommendations = [recommendations[0]]
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
