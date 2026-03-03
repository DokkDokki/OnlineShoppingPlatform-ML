import numpy as np
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

# --- 1. SETUP ---
st.set_page_config(page_title="Paul's Shop", layout="wide")

# --- 2. SMART DATA LOADER ---
@st.cache_data
def load_data():
    if not os.path.exists('data/products.csv') or not os.path.exists('data/transactions.csv'):
        st.error("⚠️ CSV files not found in /data folder!")
        return pd.DataFrame(), pd.DataFrame()
    
    products = pd.read_csv('data/products.csv')
    transactions = pd.read_csv('data/transactions.csv')

    # FIX: Handle missing 'purchase_date'
    if 'purchase_date' not in transactions.columns:
        # Check for Kaggle's 'InvoiceDate'
        if 'InvoiceDate' in transactions.columns:
            transactions = transactions.rename(columns={'InvoiceDate': 'purchase_date'})
        else:
            transactions['purchase_date'] = pd.to_datetime("2024-01-01")
    
    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    return products, transactions

# Assign to consistent names
df_products, df_transactions = load_data()

# --- 3. ML MODEL ---
if not df_products.empty:
    # Use 'df_products' consistently to fix NameError: 'df_p'
    df_products['features'] = df_products['category'].astype(str) + " " + df_products['product_name'].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_products['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- 4. EVALUATION FUNCTION ---
def evaluate_recommender(df, sim, k=5, threshold=0.3):
    y_true, y_pred, rr = [], [], []
    sample = np.random.choice(len(df), min(len(df), 50), replace=False)
    for idx in sample:
        scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)[1:k+1]
        has_rel = 1 if (np.sum(sim[idx] > threshold) - 1) > 0 else 0
        found = 0
        for rank, (r_idx, score) in enumerate(scores, 1):
            is_rel = 1 if score >= threshold else 0
            y_true.append(has_rel); y_pred.append(is_rel)
            if is_rel and found == 0: found = 1/rank
        rr.append(found)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return p, r, np.mean(rr)

# --- 5. ADMIN LOGGING ---
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

# Sidebar Logic
with st.sidebar:
    if not st.session_state['admin_logged_in']:
        u = st.text_input("Username")
        p_input = st.text_input("Password", type="password")
        if st.button("Login"):
            if u == "admin" and p_input == "1234":
                st.session_state['admin_logged_in'] = True
                st.rerun()
    else:
        if st.button("Log Out"):
            st.session_state['admin_logged_in'] = False
            st.rerun()

# --- 6. MAIN UI ---
st.title("🛍️ Paul's Shop")
t1, t2 = st.tabs(["🛒 AI Recommender", "📈 Business Dashboard"])

with t1:
    query = st.text_input("Search:", "Red ceramic mug")
    budget = st.slider("Budget", 0, 500, 100)
    if st.button("Get Recommendations"):
        vec = tfidf.transform([query])
        sim_scores = cosine_similarity(vec, tfidf_matrix).flatten()
        df_products['score'] = sim_scores
        res = df_products[df_products['price'] <= budget].sort_values('score', ascending=False).head(6)
        cols = st.columns(3)
        for i, row in enumerate(res.itertuples()):
            with cols[i % 3]:
                st.info(f"**{row.product_name}**\n\nMatch: {row.score:.1%}")

with t2:
    if st.session_state.get('admin_logged_in'):
        # FIX: Define precision, recall, mrr before using them
        precision, recall, mrr = evaluate_recommender(df_products, cosine_sim)
        
        st.subheader("📊 Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@5", f"{precision:.2f}")
        c2.metric("Recall@5", f"{recall:.2f}")
        c3.metric("MRR", f"{mrr:.2f}")
        
        st.divider()
        st.write("### Sales Analysis")
        # FIX: Check for 'product_id' before merging
        if 'product_id' in df_transactions.columns and 'product_id' in df_products.columns:
            merged = df_transactions.merge(df_products[['product_id', 'category']], on='product_id')
            st.bar_chart(merged.groupby('category')['amount'].sum())
    else:
        st.warning("Please log in via the sidebar to view metrics.")