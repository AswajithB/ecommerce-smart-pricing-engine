
# E-Commerce Smart Pricing Engine 🏷️

![Project Banner](https://img.shields.io/badge/Machine%20Learning-Price%20Prediction-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python)

## 📌 Project Overview

**E-Commerce Smart Pricing Engine** is an intelligent web application designed to help e-commerce sellers determine the optimal selling price for their products. By leveraging machine learning algorithms trained on historical sales data, the system predicts accurate price points based on product details such as category, brand, condition, and text descriptions.

Beyond simple prediction, the application includes a **Dynamic Pricing Engine** that adjusts recommendations based on simulated real-time market factors like seasonality, brand trends, and market demand.

## 🚀 Key Features

*   **🤖 AI-Powered Price Prediction:** Uses a **Ridge Regression** model to predict product prices with high accuracy based on diverse input features.
*   **📈 Dynamic Pricing Algorithm:** Automatically adjusts base price predictions for:
    *   **Seasonality:** (e.g., Winter coats fetch higher prices in winder).
    *   **Market Trends:** (e.g., Trending tech items).
    *   **Brand Premiums:** (e.g., Apple, Nike).
*   **💱 Real-Time Currency Conversion:** Automatically converts predicted prices from USD to INR using live exchange rates.
*   **📊 Interactive Analytics:**
    *   **Price Optimization:** Visualizes the trade-off between price, potential profit, and customer satisfaction.
    *   **Market Distribution:** Shows where the item stands compared to similar products in the market.
*   **📦 Inventory Management:** A dedicated dashboard to view, search, and manage product listings with automated price suggestions.

## 🛠️ Tech Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** Scikit-learn, Pandas, NumPy, SciPy
*   **Data Processing:** NLTK (Text processing), Joblib (Model persistence)
*   **Frontend:** HTML5, CSS3, JavaScript (Bootstrap/Custom)
*   **Visualization:** Matplotlib/Seaborn (for analysis notebooks), Chart.js (implied for web dashboards)

## 🧠 Machine Learning Pipeline

The core of this project is a robust ML pipeline that processes mixed data types:

1.  **Data Preprocessing:**
    *   **Text Data:** TF-IDF Vectorization for item descriptions.
    *   **Categorical Data:** One-Hot Encoding and Label Binarization for brands and categories.
    *   **Numerical Data:** Standardization and Log-transformation of target variables (`log1p`).

2.  **Model Evaluation:**
    We experimented with several models. The **Ridge Regression** model was selected for deployment due to its balance of speed and accuracy.

    | Model | RMSLE Score | Status |
    | :--- | :--- | :--- |
    | **Ridge Regression** | **0.475** | ✅ **Deployed** |
    | Linear Regression | 0.475 | |
    | Random Forest | 0.719 | |

    *Metric: Root Mean Squared Logarithmic Error (RMSLE)*

## 📂 Project Structure

```
├── app.py                      # Main Flask application
├── Price_recommendation.ipynb  # Model training & analysis notebook
├── requirements.txt            # Project dependencies
├── dataset/                    # Training and test datasets
├── static/                     # CSS, JS, and images
├── templates/                  # HTML Templates (index.html, inventory.html)
└── *.joblib                    # Serialized ML models and transformers
```

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ecommerce-smart-pricing-engine.git
    cd price-recommendation
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    python app.py
    ```

5.  **Access the Dashboard:**
    Open your browser and navigate to `http://127.0.0.1:5000/`.

## 🔮 Future Improvements

*   **Deep Learning:** Implementing LSTM or BERT for better text understanding of product descriptions.
*   **Image Analysis:** Using CNNs to incorporate product images into the pricing model.
*   **User Accounts:** Allowing sellers to save their inventory and track price changes over time.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
*Created by Aswaj*
