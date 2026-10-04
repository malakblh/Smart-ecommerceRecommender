import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart E-commerce Recommender",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv("data/cleaned_retail.csv")
    # Remove known non-product entry
    data = data[data["StockCode"] != "BANK CHARGES"]
    return data


df = load_data()

# StockCode -> Description lookup, so the UI can show
# readable product names instead of raw codes
product_names = (
    df[["StockCode", "Description"]]
    .dropna()
    .drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛒 Smart Recommender")

    st.markdown("---")

    st.subheader("📌 About")

    st.write(
        """
        This application recommends products based on
        historical customer purchasing behavior.
        """
    )

    st.subheader("🧠 Model")

    st.markdown(
        """
        **Method:** Item-Item Collaborative Filtering

        **Similarity:** Cosine Similarity

        **Data:** Online Retail transactions

        **Recommendation:** Top-N similar products
        """
    )

    st.subheader("🛠️ Technologies")

    st.markdown(
        """
        - Python
        - Pandas
        - Scikit-learn
        - Streamlit
        """
    )


# ============================================================
# CREATE CUSTOMER-PRODUCT MATRIX
# ============================================================

@st.cache_data
def create_recommendation_model(data):

    user_item_binary = (
        data.assign(Purchased=1)
        .pivot_table(
            index="CustomerID",
            columns="StockCode",
            values="Purchased",
            aggfunc="max",
            fill_value=0
        )
    )

    binary_similarity = cosine_similarity(user_item_binary.T)

    binary_sim_df = pd.DataFrame(
        binary_similarity,
        index=user_item_binary.columns,
        columns=user_item_binary.columns
    )

    return binary_sim_df


item_sim_df = create_recommendation_model(df)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_products(product_code, top_n=5):

    if product_code not in item_sim_df.columns:
        return pd.DataFrame(
            columns=["StockCode", "Product", "Similarity"]
        )

    similar_products = (
        item_sim_df[product_code]
        .sort_values(ascending=False)
        .iloc[1:top_n + 1]
    )

    recommendations = pd.DataFrame({
        "StockCode": similar_products.index,
        "Similarity": similar_products.values
    })

    recommendations["Product"] = (
        recommendations["StockCode"].map(product_names)
    )

    return recommendations[
        ["StockCode", "Product", "Similarity"]
    ].reset_index(drop=True)


# ============================================================
# HEADER
# ============================================================

st.title("🛒 Smart E-commerce Recommender")

st.markdown(
    """
    ### 🤖 Discover products you may like

    This recommendation system analyzes historical customer
    purchasing behavior and recommends products with similar
    purchasing patterns.
    """
)


# ============================================================
# PROJECT STATISTICS
# ============================================================

total_customers = df["CustomerID"].nunique()
total_products = df["StockCode"].nunique()
total_transactions = df["InvoiceNo"].nunique()
total_quantity = df["Quantity"].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "🛍️ Products",
        f"{total_products:,}"
    )

with col3:
    st.metric(
        "🧾 Transactions",
        f"{total_transactions:,}"
    )

with col4:
    st.metric(
        "📦 Items Purchased",
        f"{total_quantity:,}"
    )


st.divider()


# ============================================================
# PRODUCT SELECTION
# ============================================================

st.subheader("🔎 Find Similar Products")

product_codes = sorted(item_sim_df.columns.tolist())
selected_product = st.selectbox(
    "Choose a product:",
    product_codes,
    format_func=lambda code: product_names.get(code, code)
)


number = st.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=10,
    value=5
)

# ============================================================
# SELECTED PRODUCT INFORMATION
# ============================================================

selected_product_data = df[
    df["StockCode"] == selected_product
]

product_customers = selected_product_data["CustomerID"].nunique()
product_quantity = selected_product_data["Quantity"].sum()
product_transactions = selected_product_data["InvoiceNo"].nunique()

st.subheader("📊 Selected Product")
st.caption(product_names.get(selected_product, selected_product))

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "👥 Customers",
        f"{product_customers:,}"
    )

with info2:
    st.metric(
        "📦 Quantity Purchased",
        f"{product_quantity:,}"
    )

with info3:
    st.metric(
        "🧾 Transactions",
        f"{product_transactions:,}"
    )

# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button(
    "✨ Get Recommendations",
    use_container_width=True
):

    recommendations = recommend_products(
        selected_product,
        number
    )

    st.subheader("🤖 Recommended Products")

    if recommendations.empty:

        st.warning(
            "No recommendations found for this product."
        )

    else:

        for index, row in recommendations.iterrows():

            product = row["Product"]
            similarity = float(row["Similarity"])

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f"### 🛍️ {product}"
                )

            with col2:
                st.metric(
                    "Similarity",
                    f"{similarity:.3f}"
                )

            st.progress(
                min(similarity, 1.0)
            )

            if index < len(recommendations) - 1:
                st.divider()


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.divider()

st.subheader("💡 How to use")

st.markdown(
    """
    1. 🔎 Select a product from the dropdown.
    2. 🎚️ Choose how many recommendations you want.
    3. ✨ Click **Get Recommendations**.
    4. 🛍️ Explore products with similar customer purchasing patterns.
    """
)

st.subheader("🧠 How does it work?")
st.info(
    """
    💡 The similarity score represents similarity between
    customer purchasing patterns. It does not represent the
    probability that a customer will purchase the recommended product.
    """
)
st.markdown(
    """
    The system uses **item-item collaborative filtering**.

    1. Customer purchasing behavior is converted into a
       customer-product matrix.
    2. Each product is represented by the customers who
       purchased it.
    3. **Cosine similarity** is used to compare products.
    4. The most similar products are returned as recommendations.

    The model focuses on whether a customer purchased a product,
    rather than the quantity purchased.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart E-commerce Recommender • "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)