import pandas as pd
import random
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NUM_PRODUCTS = 50
NUM_TRANSACTIONS = 500
CATEGORIES = ['Electronics', 'Fashion', 'Home & Living', 'Sports', 'Beauty']

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# --- 1. GENERATE PRODUCTS (For the Questionnaire) ---
def generate_products():
    products = []
    for i in range(1, NUM_PRODUCTS + 1):
        category = random.choice(CATEGORIES)
        
        # Assign realistic price ranges
        if category == 'Electronics':
            price = random.randint(100, 2000)
            names = ['Smartphone', 'Laptop', 'Headphones', 'Smart Watch', 'Camera']
        elif category == 'Fashion':
            price = random.randint(20, 200)
            names = ['T-Shirt', 'Jeans', 'Sneakers', 'Jacket', 'Dress']
        else:
            price = random.randint(10, 100)
            names = ['Item', 'Tool', 'Accessory', 'Decor', 'Gadget']
            
        products.append({
            'product_id': i,
            'product_name': f"{random.choice(['Pro', 'Basic', 'Super', 'Ultra'])} {random.choice(names)} {i}",
            'category': category,
            'price': price,
            'rating': round(random.uniform(3.0, 5.0), 1),
            'tags': f"{category}, {random.choice(['Budget', 'Premium'])}, {random.choice(['New', 'Bestseller'])}"
        })
    return pd.DataFrame(products)

# --- 2. GENERATE TRANSACTIONS (For Trend Analysis) ---
def generate_transactions(product_df):
    transactions = []
    end_date = datetime.now()
    
    for i in range(NUM_TRANSACTIONS):
        # Random date within last year
        random_days = random.randint(0, 365)
        date = end_date - timedelta(days=random_days)
        
        # Pick a random product
        product = product_df.sample(1).iloc[0]
        
        transactions.append({
            'transaction_id': i + 1,
            'user_id': random.randint(100, 150),
            'product_id': product['product_id'],
            'purchase_date': date.strftime('%Y-%m-%d'),
            'amount': product['price'],
            'quantity': random.randint(1, 3)
        })
    return pd.DataFrame(transactions)

# --- SAVE DATA ---
if __name__ == "__main__":
    print("Generating data...")
    df_products = generate_products()
    df_transactions = generate_transactions(df_products)
    
    # Save to the 'data' folder
    df_products.to_csv('data/products.csv', index=False)
    df_transactions.to_csv('data/transactions.csv', index=False)
    
    print("Success! Created 'data/products.csv' and 'data/transactions.csv'")