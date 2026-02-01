import pandas as pd

# 1. Load the data
print("📂 Loading data...")
try:
    products = pd.read_csv('data/products.csv')
    transactions = pd.read_csv('data/transactions.csv')
except FileNotFoundError:
    print("❌ Error: Files not found. Did you run 'convert_retail.py'?")
    exit()

# 2. EXPLORE PRODUCTS
print("\n" + "="*40)
print("📦 PRODUCTS DATASET SUMMARY")
print("="*40)
print(f"Total Products: {len(products)}")
print(f"Columns: {list(products.columns)}")
print("\n--- First 3 Rows ---")
print(products.head(3))
print("\n--- Missing Values? ---")
print(products.isnull().sum())
print("\n--- Price Stats ---")
print(products['price'].describe())
print(f"\nUnique Categories: {products['category'].nunique()}")
print(f"Example Categories: {products['category'].unique()[:5]}")


# 3. EXPLORE TRANSACTIONS
print("\n" + "="*40)
print("💳 TRANSACTIONS DATASET SUMMARY")
print("="*40)
print(f"Total Transactions: {len(transactions)}")
print("\n--- First 3 Rows ---")
print(transactions.head(3))
print("\n--- Date Range ---")
# Convert to datetime to find min/max easily
transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
print(f"Start Date: {transactions['purchase_date'].min()}")
print(f"End Date:   {transactions['purchase_date'].max()}")

print("\n" + "="*40)
print("✅ EXPLORATION COMPLETE")