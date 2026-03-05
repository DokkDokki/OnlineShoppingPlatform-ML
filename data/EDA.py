import pandas as pd

# 1. Load your dataset
# Make sure the file name matches your actual CSV file
df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

print("="*50)
print("              DATASET OVERVIEW")
print("="*50)

# 2. General Info (Source/Size)
print(f"Source: Kaggle (Online Retail Dataset)")
print("\nDataset Shape:", df.shape)
# 3. Feature Types
print("\nFEATURE TYPES:")
print(df.dtypes)
print("-" * 50)

# 4. Target Variable Explanation
# In recommenders, the target is calculated similarity, not a column.
print("\nTARGET VARIABLE:")
print("Type: Dynamic (Similarity Score)")
print("Definition: Cosine Similarity between Product and Query vectors.")
print("-" * 50)

# 5. Data Preview
print("DATA (First 5 Rows):")
print(df.head())
print("="*50)