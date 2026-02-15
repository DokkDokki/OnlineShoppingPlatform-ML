import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score


# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Paul's Shop", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        products = pd.read_csv('data/products.csv')
        transactions = pd.read_csv('data/transactions.csv')
        transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
        return products, transactions
    except FileNotFoundError:
        st.error("Data not found! Run 'convert_retail.py' first.")
        return pd.DataFrame(), pd.DataFrame()

df_products, df_transactions = load_data()

# --- 3. TRAIN ML MODEL (Content-Based) ---
if not df_products.empty:
    # Combine Category + Name to create a "Feature Soup" for the AI to read
    df_products['features'] = df_products['category'] + " " + df_products['product_name']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_products['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- [MODEL EVALUATION] ---
def evaluate_recommender(df_products, cosine_sim, k=5):
    y_true = []
    y_pred = []

    for idx in range(len(df_products)):

        # Get similarity scores for each product
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort by similarity (exclude itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]

        recommended_indices = [i[0] for i in sim_scores]

        query_category = df_products.iloc[idx]['category']

        for rec_idx in recommended_indices:
            recommended_category = df_products.iloc[rec_idx]['category']

            # Relevant if same category
            y_true.append(1)
            y_pred.append(1 if recommended_category == query_category else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


precision, recall, f1 = evaluate_recommender(df_products, cosine_sim)

print("Precision@K:", precision)
print("Recall@K:", recall)
print("F1-score@K:", f1)


# --- 4. SIDEBAR (Login & Admin Panel) ---
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

with st.sidebar:
    st.header("⚙️ Admin Panel")
    
    # IF NOT LOGGED IN -> SHOW LOGIN FORM
    if not st.session_state['admin_logged_in']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state['admin_logged_in'] = True
                st.rerun() # Force reload to show the new menu
            else:
                st.error("❌ Invalid credentials")
    
    # IF LOGGED IN -> SHOW ADMIN TOOLS
    else:
        st.success("✅ Logged in as Admin")
        if st.button("Log Out"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
            
        st.divider()
        
        st.subheader("➕ Add New Product")
        with st.form("add_product_form"):
            new_name = st.text_input("Product Name")
            new_category = st.selectbox("Category", ['Gifts', 'Kitchen', 'Home Decor', 'Office', 'Accessories'])
            new_price = st.number_input("Price ($)", min_value=0.0, step=0.01)
            
            submitted = st.form_submit_button("Add to Database")
            if submitted and new_name:
                new_id = df_products['product_id'].max() + 1
                new_row = pd.DataFrame({
                    'product_id': [new_id],
                    'product_name': [new_name],
                    'category': [new_category],
                    'price': [new_price],
                    'rating': [5.0]
                })
                new_row.to_csv('data/products.csv', mode='a', header=False, index=False)
                st.success(f"Added '{new_name}'!")

# --- 5. MAIN APP ---
st.title("🛍️ Paul's Shop")
st.markdown("Welcome to Paul's Shop!")

tab1, tab2 = st.tabs(["🛒 AI Recommender", "📈 Business Dashboard"])

# ==========================================
# TAB 1: RECOMMENDATION ENGINE
# ==========================================
with tab1:
    st.subheader("Find Your Perfect Product")
    
    # NEW LAYOUT: Category Filter -> Product Select
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Tell us what you're looking for:", "Red ceramic mug")
    with col2:
        budget = st.slider("💰 Max Budget ($)", 0, 100, 50)

    # THE RECOMMENDATION LOGIC
    if st.button("✨ Get Recommendations", type="primary"):
        if search_query:
            # Vectorize the search query
            query_vec = tfidf.transform([search_query])
            
            # Calculate similarity between query and all products
            query_sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Get top 6 similar products
            # Use argsort to get indices that would sort the array, then reverse for descending order
            sim_indices = query_sim_scores.argsort()[:-7:-1] # Get top 6 indices (excluding self-similarity if query was a product)
            
            results = df_products.iloc[sim_indices].copy()
            
            # 4. Filter by Budget
            results = results[results['price'] <= budget]
            
            if not results.empty:
                st.success(f"Found {len(results)} items similar to your search: '{search_query}':")
                
                # Display results in columns
                cols = st.columns(3)
                for idx, row in enumerate(results.itertuples()):
                    with cols[idx % 3]:
                        st.image(f"https://via.placeholder.com/300x200.png?text={row.category}", width='stretch')
                        st.subheader(row.product_name)
            else:
                st.info(f"No recommendations found for '{search_query}' within your budget.")
        else:
            st.warning("Please enter a product description to get recommendations.")

# ==========================================
# TAB 2: BUSINESS DASHBOARD (Protected)
# ==========================================
with tab2:
    # 1. CHECK LOGIN STATUS
    # We use .get() to avoid errors if the key doesn't exist yet
    if st.session_state.get('admin_logged_in', False):
        
        # --- SHOW DASHBOARD (Logged In) ---
        st.subheader("📊 Sales Analytics")
        
        # KPI Row
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Revenue", f"${df_transactions['amount'].sum():,.2f}")
        kpi2.metric("Total Orders", len(df_transactions))
        kpi3.metric("Avg Order Value", f"${df_transactions['amount'].mean():.2f}")
        
        st.divider()


        # --- [NEW: SHOW MODEL METRICS IN DASHBOARD] ---
        st.subheader("🤖 Recommender Model Evaluation")
        st.write(f"**Precision@5:** {precision:.2f}")
        st.write(f"**Recall@5:** {recall:.2f}")
        st.write(f"**F1-score@5:** {f1:.2f}")
        
        st.divider()
     

        chart1, chart2 = st.columns(2)
        
        with chart1:
            st.write("### Best Selling Categories")
            merged = df_transactions.merge(df_products[['product_id', 'category']], on='product_id')
            cat_sales = merged.groupby('category')['amount'].sum().sort_values(ascending=False)
            st.bar_chart(cat_sales)
            
        with chart2:
            st.write("### Revenue Over Time")
            monthly_sales = df_transactions.set_index('purchase_date').resample('M')['amount'].sum()
            st.line_chart(monthly_sales)
            
    # 2. IF NOT LOGGED IN
    else:
        # --- SHOW LOCK SCREEN (Logged Out) ---
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.warning("🔒 **Admin Access Required**")
            st.info("Please log in using the sidebar to view sensitive business data.")
            # Optional: Add a lock icon image
            st.image("https://cdn-icons-png.flaticon.com/512/6146/6146689.png", width=100)