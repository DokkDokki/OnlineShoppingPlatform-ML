import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# =========================
# Load Data
# =========================

products = pd.read_csv('data/products.csv')
transactions = pd.read_csv('data/transactions.csv')
transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])

df= pd.merge(
    transactions,
    products,
    left_on = "product_id",
    right_on = "product_id",
    how = "inner"
)

@st.cache_data
def load_data():
    df["interaction"] = np.log1p(df["amount"])
    return df

df = load_data()

st.title("🛍️ E-Commerce Recommendation System")

# =========================
# Build User-Item Matrix
# =========================

user_ids = df["user_id"].unique()
item_ids = df["product_id"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}
inv_item_map = {v: k for k, v in item_map.items()}

rows = df["user_id"].map(user_map)
cols = df["product_id"].map(item_map)
values = df["interaction"]

matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(item_ids))
)

# =========================
# Train Matrix Factorization
# =========================

@st.cache_resource
def train_model(_matrix):

    svd = TruncatedSVD(n_components=50, random_state=42)
    user_embeddings = svd.fit_transform(matrix)
    item_embeddings = svd.components_.T
    return user_embeddings, item_embeddings

user_emb, item_emb = train_model(matrix)

# =========================
# Recommendation Function
# =========================

def recommend(user_id, top_k=10):
    if user_id not in user_map:
        return pd.DataFrame()

    user_idx = user_map[user_id]
    scores = user_emb[user_idx] @ item_emb.T

    top_items = np.argsort(-scores)[:top_k]
    product_ids = [inv_item_map[i] for i in top_items]

    return df[df["product_id"].isin(product_ids)][
        ["product_id", "product_name", "category", "price"]
    ].drop_duplicates()

# =========================
# Streamlit UI
# =========================

selected_user = st.selectbox(
    "Select User ID",
    user_ids
)

if st.button("Get Recommendations"):
    recs = recommend(selected_user)

    if recs.empty:
        st.write("No recommendations available.")
    else:
        st.subheader("Top Recommendations")
        st.dataframe(recs)


