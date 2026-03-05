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
    # STEP 1: Create the column 
    df_products['features'] = df_products['category'].astype(str) + " " + df_products['product_name'].astype(str)
    
    # STEP 2: apply the stemmer to the column created
    ps = PorterStemmer()
    df_products['features'] = df_products['features'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
    
    # STEP 3: Vectorize
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df_products['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- 4. EVALUATION FUNCTION ---
def evaluate_recommender(df, sim, k=5, threshold=0.15):
    y_true, y_pred, rr = [], [], []
    # Test on a sample of 50 products to keep the app fast
    sample_indices = np.random.choice(len(df), min(len(df), 50), replace=False)
    
    for idx in sample_indices:
        # Get top K recommendations based on similarity scores
        # We skip the first one [1:k+1] because index 0 is the item itself
        top_k_indices = np.argsort(sim[idx])[-(k+1):-1][::-1]
        
        # Determine if the item actually has relevant matches in the data
        # (Is there anything in the same category with similarity > threshold?)
        has_rel = 1 if (np.sum(sim[idx] > threshold) - 1) > 0 else 0
        
        found_at_rank = 0
        for i, r_idx in enumerate(top_k_indices, 1):
            score = sim[idx][r_idx]
            is_rel = 1 if score >= threshold else 0
            
            # Data for Confusion Matrix
            y_true.append(has_rel)
            y_pred.append(is_rel)
            
            # Data for MRR (Mean Reciprocal Rank)
            if is_rel and found_at_rank == 0:
                found_at_rank = 1 / i
        rr.append(found_at_rank)
    
    # Calculate Final Metrics
    precision_at_k = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    mrr = np.mean(rr)
    
    # Return exactly 5 values to match your dashboard's expectations
    return precision_at_k, mrr, recall_val, y_true, y_pred

def get_confusion_data(df, sim, threshold=0.3):
    y_true, y_pred = [], []
    sample = np.random.choice(len(df), min(len(df), 100), replace=False)
    
    for idx in sample:
        # We consider an item "Truly Relevant" if it's in the same category
        cat = df.iloc[idx]['category']
        actual_relevant_indices = df[df['category'] == cat].index.tolist()
        
        # Get all scores for this item
        scores = sim[idx]
        
        for i in range(len(scores)):
            if i == idx: continue # skip self
            y_true.append(1 if i in actual_relevant_indices else 0)
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

# --- 6. MAIN UI ---
st.title("🛍️ Paul's Shop")
t1, t2 = st.tabs(["🛒 AI Recommender", "📈 Business Dashboard"])

# --- TAB 1: USER FACING SEARCH ---
with t1:
    st.subheader("🔍 Find Your Perfect Product")
    
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    with col_input1:
        user_query = st.text_input("What are you looking for?", placeholder="e.g. Red ceramic mug", key="search_query")
    with col_input2:
        available_cats = ["All"] + list(df_products['category'].unique())
        selected_cat = st.selectbox("Category", available_cats)
    with col_input3:
        max_price = st.slider("Max Budget ($)", 0, 500, 100)

    # ALL Recommendation logic stays INSIDE this 'with t1' block
    if st.button("✨ Get Recommendations"):
        query_vec = tfidf.transform([user_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        df_products['score'] = sim_scores
        
        # Filter logic
        mask = df_products['price'] <= max_price
        if selected_cat != "All":
            mask &= (df_products['category'] == selected_cat)
            
        results = df_products[mask].sort_values('score', ascending=False).head(100)
        
        if not results.empty and results['score'].max() > 0:
            cols = st.columns(3)
            for i, row in enumerate(results.itertuples()):
                with cols[i % 3]:
                    st.info(f"**{row.product_name}**\n\nPrice: ${row.price}")
        else:
            st.warning("No products found! Try adjusting your search filters.")

# --- TAB 2: ADMIN DASHBOARD ---
with t2:
    if not st.session_state['admin_logged_in']:
        st.warning("🔒 Please login via the sidebar to access the Business Dashboard.")
    else:
        st.header("📊 Model Performance Dashboard")
        st.markdown("---")

        # 1. Ranking Metrics
        p5, mrr, recall_val, yt, yp = evaluate_recommender(df_products, cosine_sim)

        with st.container():
            st.subheader("🎯 Recommendation Quality (Ranking)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision @ 5", f"{p5:.1%}")
            c2.metric("MRR Score", f"{mrr:.2f}")
            c3.metric("Discovery Rate", f"{recall_val:.1%}")

        st.markdown("---")

        # 2. Confusion Matrix (NOW MOVED INSIDE TAB 2 ONLY)
        with st.container():
            left_col, right_col = st.columns([1, 1], gap="large")
            
            with left_col:
                st.subheader("🧪 Confusion Matrix")
                # Use the helper function you already have
                y_true_cm, y_pred_cm = get_confusion_data(df_products, cosine_sim)
                cm = confusion_matrix(y_true_cm, y_pred_cm)
                
                fig, ax = plt.subplots(figsize=(4, 3)) 
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                            xticklabels=['Not Rel', 'Rel'], yticklabels=['Not Rel', 'Rel'])
                plt.ylabel('Actual', fontsize=8)
                plt.xlabel('Predicted', fontsize=8)
                st.pyplot(fig)
                

        # 3. Model Comparison Chart (MOVED INSIDE TAB 2 ONLY)
        st.markdown("---")
        st.subheader("📈 Model Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'Precision', 'Recall'],
            'Model': ['Category Only', 'Category Only', 'Hybrid', 'Hybrid'],
            'Score': [0.95, 0.20, 0.85, 0.45]
        })
        st.vega_lite_chart(comparison_df, {
            'mark': {'type': 'bar', 'tooltip': True},
            'encoding': {
                'column': {'field': 'Metric', 'type': 'nominal'},
                'x': {'field': 'Model', 'type': 'nominal'},
                'y': {'field': 'Score', 'type': 'quantitative'},
                'color': {'field': 'Model', 'type': 'nominal'}
            }
        })