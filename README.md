# 🛍️ Online Shopping Platform (ML Class Project)

A machine learning-powered e-commerce dashboard that analyzes shopping trends and provides intelligent product recommendations. This project demonstrates the use of **Content-Based Filtering** and **Trend Analysis** to support both customers and business owners.

## 👥 Group Members
* **Thapanant Khongsattra** (ID: 2420110302 DGE)
* **Pol Supmonchai** (ID: 2420110187 DGE)

## 🎯 Objectives
As defined in our project scope:
1.  **Analyze Trends:** Visualize sales data to understand popular products and peak shopping periods.
2.  **Smart Recommendations:** Help undecided customers find products using an AI-driven questionnaire.
3.  **Improve Strategies:** Provide data insights to help improve advertising and inventory management.

## 🚀 Key Features

### 1. 🛒 Smart Recommender (Customer View)
* **Knowledge-Based Engine:** A "Questionnaire" system that filters products based on user budget and specific needs (solving the "Cold Start" problem).
* **Content-Based Filtering:** Uses **TF-IDF** and **Cosine Similarity** to recommend products similar to items the user already likes (e.g., *"If you like this 'Heart Tea Light', you might also like..."*).

### 2. 📈 Business Dashboard (Admin View)
* **Sales Trends:** Line charts showing revenue fluctuations over time.
* **Category Analysis:** Bar charts identifying best-selling product categories.
* **KPI Metrics:** Real-time tracking of Total Revenue and Total Orders.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Frontend:** Streamlit (Web App Framework)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Feature Extraction, Pairwise Metrics)
* **Visualization:** Matplotlib

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/DokkDokki/OnlineShoppingPlatform-ML.git](https://github.com/DokkDokki/OnlineShoppingPlatform-ML.git)
cd OnlineShoppingPlatform-ML
```
### 2. Create the virtual environment
# Windows
1. ```python -m venv venv```

2. ```.\venv\Scripts\activate```

# Mac/Linux
1. ```python3 -m venv venv```

2. ```source venv/bin/activate```
### 3. Install required Libraries
```pip install -r requirements.txt```

## Data Setup
1. Download the Dataset: https://www.kaggle.com/datasets/carrie1/ecommerce-data

    - Go to the E-Commerce Data (UCI) on Kaggle.

    - Download the file.

    - Rename the file to data.csv.

2. Place in Project:

    - Create a folder named data inside the project directory.

    - Move data.csv into that folder.

3. Run the Converter: We use a script to clean the data and simulate "Categories" and "Ratings" (since the raw dataset lacks them).

`python convert_retail.py`

You should see a message: Success! Dataset converted.

## How to run

``streamlit run app.py``