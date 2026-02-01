import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Smart Shop AI", layout="wide")
st.title("🛍️ Smart Shopping Platform")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        products = pd.read_csv('data/products.csv')
        transactions = pd.read_csv('data/transactions.csv')
        # Convert date column to datetime objects for math
        transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
        return products, transactions
    except FileNotFoundError:
        st.error("Data file not found. Run generate_data.py first!")
        return pd.DataFrame(), pd.DataFrame()

df_products, df_transactions = load_data()

# --- 3. CREATE TABS ---
# We split the app into two parts as per your project scope
tab1, tab2 = st.tabs(["🛒 Shopper Assistant", "📈 Business Dashboard"])

# ==========================================
# TAB 1: SHOPPER ASSISTANT (Questionnaire)
# ==========================================
with tab1:
    st.header("Find Your Perfect Product")
    
    # Sidebar moved inside the tab logic conceptually
    st.sidebar.header("Filter Options")
    target_category = st.sidebar.selectbox("Category", df_products['category'].unique())
    budget = st.sidebar.slider("Max Budget ($)", int(df_products['price'].min()), int(df_products['price'].max()), 500)
    
    # Recommendation Logic (Same as before)
    def get_recommendations(df, category, max_price):
        rec = df[df['category'] == category].copy()
        rec['score'] = 0
        rec.loc[rec['price'] <= max_price, 'score'] += 50
        rec['score'] += rec['rating'] * 10
        return rec[rec['price'] <= max_price * 1.2].sort_values('score', ascending=False)

    results = get_recommendations(df_products, target_category, budget)
    
    if not results.empty:
        # Show Top 3 Cards
        cols = st.columns(3)
        for idx, row in enumerate(results.head(3).itertuples()):
            with cols[idx]:
                st.subheader(row.product_name)
                st.write(f"**${row.price}** | ⭐ {row.rating}")
                st.progress(int(row.score)/100) # Visual score bar
    else:
        st.info("No matches found. Adjust your budget.")

# ==========================================
# TAB 2: BUSINESS DASHBOARD (Trend Analysis)
# ==========================================
with tab2:
    st.header("📊 Market Trends & Analytics")
    st.write("Analyze sales trends to improve advertising strategies.")

    # KPI Metrics
    total_sales = df_transactions['amount'].sum()
    total_orders = len(df_transactions)
    col1, col2 = st.columns(2)
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Total Orders", total_orders)

    # --- CHART 1: Monthly Sales Trend ---
    st.subheader("💰 Monthly Sales Trend")
    # Group data by Month (M) and sum the amount
    monthly_sales = df_transactions.set_index('purchase_date').resample('M')['amount'].sum()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_sales.index, monthly_sales.values, marker='o', color='green', linestyle='--')
    ax.set_title("Revenue over Time")
    ax.set_ylabel("Sales ($)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.caption("Observation: Identify peaks to plan ads for specific months.")

    # --- CHART 2: Top Selling Categories ---
    st.subheader("🏆 Best Selling Categories")
    # Merge transactions with products to get category names
    merged_df = pd.merge(df_transactions, df_products, on='product_id')
    category_sales = merged_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    st.bar_chart(category_sales)