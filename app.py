import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Smart E-commerce Recommender",
    page_icon="🛒",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("🛒 Smart E-commerce Recommender")

st.write(
    "Select a product to discover similar products "
    "based on customer purchasing behavior."
)


# -----------------------------------
# Load cleaned dataset
# -----------------------------------

df = pd.read_csv("data/cleaned_retail.csv")


# -----------------------------------
# Create User-Item Matrix
# -----------------------------------

user_item_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)


# -----------------------------------
# Calculate Product Similarity
# -----------------------------------

item_similarity = cosine_similarity(user_item_matrix.T)

item_sim_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)


# -----------------------------------
# Product Selection
# -----------------------------------

products = item_sim_df.columns.tolist()

selected_product = st.selectbox(
    "🔎 Choose a product:",
    products
)


# -----------------------------------
# Number of Recommendations
# -----------------------------------

number = st.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=10,
    value=5
)


# -----------------------------------
# Recommendation Function
# -----------------------------------

def recommend_products(product_name, top_n=5):

    similar_products = (
        item_sim_df[product_name]
        .sort_values(ascending=False)
    )

    recommendations = similar_products.iloc[1:top_n + 1]

    return recommendations


# -----------------------------------
# Recommendation Button
# -----------------------------------

if st.button("✨ Get Recommendations"):

    recommendations = recommend_products(
        selected_product,
        number
    )

    st.subheader("Recommended Products")

    for product, similarity in recommendations.items():

        st.write(
            f"🛍️ **{product}** — "
            f"Similarity: {similarity:.2f}"
        )