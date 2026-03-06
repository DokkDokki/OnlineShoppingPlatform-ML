import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv('data/products.csv', encoding='ISO-8859-1')

# Diagnostic: Check columns and data types
print("Columns found in CSV:", df.columns.tolist())

# 2. Preprocessing
# Ensure we use a copy to avoid SettingWithCopy warnings
clean_df = df.dropna(subset=['Description', 'CustomerID']).copy()
clean_df = clean_df[clean_df['Quantity'] > 0]
clean_df['TotalPrice'] = clean_df['Quantity'] * clean_df['UnitPrice']
clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])

print(f"Rows remaining after cleaning: {len(clean_df)}")

# 3. EDA Visualizations

plt.figure(figsize=(10, 6))
top_products = clean_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', color='skyblue')
plt.title('Top 10 Products by Quantity Sold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


clean_df['Month'] = clean_df['InvoiceDate'].dt.to_period('M')
monthly_revenue = clean_df.groupby('Month')['TotalPrice'].sum()
plt.figure(figsize=(10, 5))
monthly_revenue.astype(float).plot(kind='line', marker='o', color='darkorange')
plt.title('Total Revenue Trends Over Time')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()

# 4. EDA Table
print("\n--- Top 10 Customers by Revenue ---")
print(clean_df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10))