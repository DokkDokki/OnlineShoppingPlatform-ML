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

with t1:
    st.subheader("🔍 Find Your Perfect Product")
    
    # 1. USER INPUT SECTION
    # We use columns to make the input area look organized
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    
    with col_input1:
        # Text Input: This becomes the 'query' for the AI
        user_query = st.text_input("What are you looking for?", placeholder="e.g. Red ceramic mug")
        
    with col_input2:
        # Selectbox: Let user filter by a specific category
        available_cats = ["All"] + list(df_products['category'].unique())
        selected_cat = st.selectbox("Category", available_cats)
        
    with col_input3:
        # Slider: Set a price limit
        max_price = st.slider("Max Budget ($)", 0, 500, 100)

    # 2. TRIGGER BUTTON
if st.button("✨ Get Recommendations"):
    query_vec = tfidf.transform([user_query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df_products['score'] = sim_scores
    
    # Filter by budget
    results = df_products[df_products['price'] <= max_price].sort_values('score', ascending=False).head(6)
    
    if not results.empty and results['score'].max() > 0:
        cols = st.columns(3)
        for i, row in enumerate(results.itertuples()):
            with cols[i % 3]:
                st.info(f"**{row.product_name}**\n\nPrice: ${row.price}")
    else:
        # This tells you WHY nothing appeared
        st.warning("No products found! Try increasing your budget or checking your search terms.")

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

st.subheader("🧪 Confusion Matrix")

# 1. Get the data
y_true, y_pred = get_confusion_data(df_products, cosine_sim)
cm = confusion_matrix(y_true, y_pred)

# 2. CREATE A SMALLER PLOT
fig, ax = plt.subplots(figsize=(4, 3)) 

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            annot_kws={"size": 10}, # Smaller font for numbers
            cbar=False,             # Remove color bar to save space
            xticklabels=['Not Rel', 'Rel'], 
            yticklabels=['Not Rel', 'Rel'])

plt.ylabel('Actual', fontsize=8)
plt.xlabel('Predicted', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# 3. Use columns to center it or keep it to one side
col1, col2 = st.columns([1, 2]) 
with col1:
    st.pyplot(fig) 
with col2:
    st.write("**Model Accuracy Insights**")
    st.caption("This matrix shows how often the AI matches the correct category.")
    # Add a small legend/explanation
    st.markdown("""
    - **Top Left:** Correctly ignored
    - **Bottom Right:** Correct matches
    - **Top Right:** False Alarms
    """)
st.subheader("📈 Model Comparison")

# Comparison Data (Mockup based on your typical results)
comparison_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'Precision', 'Recall'],
    'Model': ['Category Only', 'Category Only', 'Hybrid (Name+Cat)', 'Hybrid (Name+Cat)'],
    'Score': [0.95, 0.20, 0.85, 0.45] # Hybrid usually has better recall but slightly lower precision
})

st.write("Comparing the 'Simple' model vs. 'Feature Soup' model:")
st.vega_lite_chart(comparison_df, {
    'mark': {'type': 'bar', 'tooltip': True},
    'encoding': {
        'column': {'field': 'Metric', 'type': 'nominal'},
        'x': {'field': 'Model', 'type': 'nominal', 'axis': {'title': ''}},
        'y': {'field': 'Score', 'type': 'quantitative'},
        'color': {'field': 'Model', 'type': 'nominal'}
    }
})