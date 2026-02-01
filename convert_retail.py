import pandas as pd
import random

print("Loading Kaggle data... (This might take a moment)")

# 1. LOAD DATA
# Try different encodings because this specific dataset often has special characters
try:
    df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    print("Error: 'data/data.csv' not found. Please rename your downloaded file!")
    exit()

# 2. CREATE PRODUCTS TABLE
# Group by Description to find unique products
products = df[['Description', 'UnitPrice']].drop_duplicates(subset='Description')
products = products.dropna() # Remove empty rows
products = products[products['UnitPrice'] > 0] # Remove free items/errors

# Rename to match our App
products = products.rename(columns={
    'Description': 'product_name', 
    'UnitPrice': 'price'
})

# --- THE FIX: ADD FAKE DATA FOR MISSING COLUMNS ---
print("Generating Categories and Ratings...")

# Create a fake ID
products['product_id'] = range(1, len(products) + 1)

# Generate Random Ratings (3.0 to 5.0)
products['rating'] = [round(random.uniform(3.0, 5.0), 1) for _ in range(len(products))]

# Generate Random Categories (Since the dataset doesn't have them)
fake_categories = ['Home Decor', 'Kitchen', 'Office', 'Gifts', 'Accessories']
products['category'] = [random.choice(fake_categories) for _ in range(len(products))]

# Reorder columns
products = products[['product_id', 'product_name', 'category', 'price', 'rating']]

# 3. CREATE TRANSACTIONS TABLE
transactions = df.copy()
transactions = transactions.dropna()
transactions = transactions[transactions['UnitPrice'] > 0]

# Merge to get the new 'product_id' we just created
transactions = transactions.merge(products[['product_name', 'product_id']], 
                                left_on='Description', 
                                right_on='product_name')

# Rename columns to match App
transactions = transactions.rename(columns={
    'InvoiceDate': 'purchase_date',
    'CustomerID': 'user_id',
    'UnitPrice': 'amount'
})

# Fix Date Format (Convert '12/1/2010 8:26' to '2010-12-01')
transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date']).dt.strftime('%Y-%m-%d')

# Handle missing user_ids (fill with 0)
transactions['user_id'] = transactions['user_id'].fillna(0).astype(int)

# Add transaction_id
transactions['transaction_id'] = range(1, len(transactions) + 1)

# Select final columns
final_transactions = transactions[['transaction_id', 'user_id', 'product_id', 'purchase_date', 'amount']]

# 4. SAVE FILES
print(f"Saving {len(products)} products and {len(final_transactions)} transactions...")
products.to_csv('data/products.csv', index=False)
final_transactions.to_csv('data/transactions.csv', index=False)

print("Success! Dataset converted.")