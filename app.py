import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

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

    print("Products Duplcates :", products.duplicated().sum())
    print("Transactions Duplcates :", products.duplicated().sum())


    # Handle missing 'purchase_date'
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


# --- 3. ML MODELS ---
@st.cache_resource 
def init_models(df):
    if df.empty:
        return None, None, None, None, None, None # Added one extra None for embeddings

    # 1. Feature Engineering
    df['features_clean'] = df['category'].astype(str) + " " + df['product_name'].astype(str)
    ps = PorterStemmer()
    df['features_stemmed'] = df['features_clean'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
    
    # 2. TF-IDF Setup 
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['features_stemmed'])
    tfidf_sim = cosine_similarity(tfidf_matrix)

    # 3. SBERT Setup 
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_embeddings = sbert_model.encode(df['features_clean'].tolist(), show_progress_bar=False)
    sbert_sim = cosine_similarity(sbert_embeddings)

    return tfidf, tfidf_matrix, tfidf_sim, sbert_model, sbert_sim, sbert_embeddings

tfidf, tfidf_matrix, tfidf_sim, sbert_model, sbert_sim, sbert_embeddings = init_models(df_products)

# --- 4. EVALUATION FUNCTION ---
def evaluate_recommender(df, sim, k=5, threshold=0.15):
    y_true, y_pred, rr = [], [], []
    # Use a fixed sample of 50 for consistent evaluation
    sample_indices = np.random.choice(len(df), min(len(df), 50), replace=False)
    
    for idx in sample_indices:
        # Get top K recommendations (skipping the item itself at index 0)
        top_k_indices = np.argsort(sim[idx])[-(k+1):-1][::-1]
        
        # Check if item has any relevant matches in the data
        has_rel = 1 if (np.sum(sim[idx] > threshold) - 1) > 0 else 0
        
        found_at_rank = 0
        for i, r_idx in enumerate(top_k_indices, 1):
            is_rel = 1 if sim[idx][r_idx] >= threshold else 0
            y_true.append(has_rel)
            y_pred.append(is_rel)
            
            # Reciprocal Rank calculation for MRR
            if is_rel and found_at_rank == 0:
                found_at_rank = 1 / i
        rr.append(found_at_rank)
    
    # Calculate Final Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    mrr = np.mean(rr)
    
    return precision, mrr, recall, y_true, y_pred

def get_confusion_data(df, sim, engine_type="TF-IDF"):
    # FIX: Initialize as two separate lists
    y_true = []
    y_pred = []
    
    sample = np.random.choice(len(df), min(len(df), 50), replace=False)
    
    # FIX: SBERT needs a much higher threshold to differentiate items
    # If using SBERT, we use the top 5% of scores as 'Relevant'
    threshold = np.percentile(sim, 95) if engine_type == "SBERT" else 0.3
    
    for idx in sample:
        actual_cat = df.iloc[idx]['category']
        actual_rel_indices = df[df['category'] == actual_cat].index.tolist()
        
        scores = sim[idx]
        for i in range(len(scores)):
            if i == idx: continue 
            y_true.append(1 if i in actual_rel_indices else 0)
            y_pred.append(1 if scores[i] >= threshold else 0)
            
    return y_true, y_pred

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

# --- 6. MAIN UI (Fixed "Not Defined" variables) ---
st.title("🛍️ Paul's Shop")
t1, t2 = st.tabs(["🛒 AI Recommender", "📈 Business Dashboard"])

with t1:
    st.subheader("🔍 Find Your Perfect Product")
    
    # 1. DEFINE the variables first using columns
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    with col_input1:
        user_query = st.text_input("What are you looking for?", placeholder="e.g. Red ceramic mug", key="search_query")
    with col_input2:
        available_cats = ["All"] + list(df_products['category'].unique())
        selected_cat = st.selectbox("Category", available_cats)
    with col_input3:
        max_price = st.slider("Max Budget ($)", 0, 500, 100)

    engine = st.radio("Select AI Brain:", ["Keyword (TF-IDF)", "Meaning (SBERT)"], horizontal=True)

    # 2. Use the defined variables inside the button logic
    if st.button("✨ Get Recommendations"):
        if not user_query:
            st.warning("Please enter a search term!")
        else:
            if engine == "Keyword (TF-IDF)":
                ps = PorterStemmer()
                query_processed = " ".join([ps.stem(word) for word in user_query.split()])
                query_vec = tfidf.transform([query_processed])
                sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            else:
                query_vec = sbert_model.encode([user_query])
                sim_scores = cosine_similarity(query_vec, sbert_embeddings).flatten()
            
            df_products['score'] = sim_scores
            
            # Apply Filters
            mask = df_products['price'] <= max_price
            if selected_cat != "All":
                mask &= (df_products['category'] == selected_cat)
            
            results = df_products[mask].sort_values('score', ascending=False).head(6)
            
            if not results.empty and results['score'].max() > 0:
                cols = st.columns(3)
                for i, row in enumerate(results.itertuples()):
                    with cols[i % 3]:
                        st.info(f"**{row.product_name}**\n\nPrice: ${row.price:.2f}")
            else:
                st.error("No matches found! Try a different search or increase your budget.")

# --- TAB 2: BUSINESS DASHBOARD (Updated for SBERT Confusion Matrix) ---
with t2:
    if st.session_state.get('admin_logged_in'):

        st.header("📊 Model Performance Dashboard")

        # ---------------- Model Selector ----------------
        eval_choice = st.radio(
            "Select AI Brain:",
            ["TF-IDF (Keyword)", "SBERT (Semantic)"],
            horizontal=True
        )

        is_sbert = "SBERT" in eval_choice
        sim_to_use = sbert_sim if is_sbert else tfidf_sim
        engine_name = "SBERT" if is_sbert else "TF-IDF"

        # ---------------- Metric Cards ----------------
        p5, mrr, recall, _, _ = evaluate_recommender(df_products, sim_to_use)

        st.subheader(f"🎯 {engine_name} Quality")

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision @ 5", f"{p5:.1%}")
        col2.metric("MRR Score", f"{mrr:.2f}")
        col3.metric("Recall", f"{recall:.1%}")

        st.divider()

        # ---------------- Calculate Metrics for Both Models ----------------
        t_p, t_mrr, t_r, _, _ = evaluate_recommender(df_products, tfidf_sim)
        s_p, s_mrr, s_r, _, _ = evaluate_recommender(df_products, sbert_sim)

        # ---------------- Precision Comparison ----------------
        st.subheader("📈 Precision Comparison")

        precision_data = pd.DataFrame([
            {"Model": "TF-IDF", "Score": t_p},
            {"Model": "SBERT", "Score": s_p}
        ])

        st.vega_lite_chart(precision_data, {
            "width": "container",
            "height": 300,
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "Model", "type": "nominal"},
                "y": {"field": "Score", "type": "quantitative", "scale": {"domain": [0, 1]}}
            }
        })

        # ---------------- Recall Comparison ----------------
        st.subheader("📈 Recall Comparison")

        recall_data = pd.DataFrame([
            {"Model": "TF-IDF", "Score": t_r},
            {"Model": "SBERT", "Score": s_r}
        ])

        st.vega_lite_chart(recall_data, {
            "width": "container",
            "height": 300,
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "Model", "type": "nominal"},
                "y": {"field": "Score", "type": "quantitative", "scale": {"domain": [0, 1]}}
            }
        })

        # ---------------- MRR Comparison ----------------
        st.subheader("📈 MRR Comparison")

        mrr_data = pd.DataFrame([
            {"Model": "TF-IDF", "Score": t_mrr},
            {"Model": "SBERT", "Score": s_mrr}
        ])

        st.vega_lite_chart(mrr_data, {
            "width": "container",
            "height": 300,
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "Model", "type": "nominal"},
                "y": {"field": "Score", "type": "quantitative", "scale": {"domain": [0, 1]}}
            }
        })
                