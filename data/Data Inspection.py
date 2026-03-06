import pandas as pd

# 1. Load your dataset
# Make sure the file name matches your actual CSV file
df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

print("="*50)
print("             DATASET OVERVIEW")
print("="*50)

# 2. General Info (Source/Size)
print(f"Source: Kaggle (Online Retail Dataset)")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 3. Feature Types
print("\nFEATURE TYPES:")
print(df.dtypes)
print("-" * 50)

# 4. MISSING VALUE ANALYSIS
print("\nMISSING VALUES:")
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100

# Creating a mini-report for columns with nulls
missing_data = pd.DataFrame({
    'Total Missing': null_counts,
    'Percentage (%)': null_percentages.round(2)
})
print(missing_data[missing_data['Total Missing'] > 0])
print("-" * 50)

# 5. Target Variable Explanation
print("\nTARGET VARIABLE:")
print("Type: Dynamic (Similarity Score)")
print("Definition: Cosine Similarity between Product and Query vectors.")
print("-" * 50)

# 6. Data Preview
print("DATA (First 5 Rows):")
print(df.head())
print("="*50)

