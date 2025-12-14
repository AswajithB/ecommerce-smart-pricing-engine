import flask
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import os
import requests

app = Flask(__name__)

# Load Artifacts
print("Loading artifacts...")
try:
    cv_name = joblib.load('cv_name.joblib')
    cv_category = joblib.load('cv_category.joblib')
    tv_description = joblib.load('tv_description.joblib')
    lb_brand = joblib.load('lb_brand.joblib')
    ohe_condition = joblib.load('ohe_condition.joblib')
    
    models = {}
    # if os.path.exists('linear_regression.joblib'):
    #     models['Linear Regression'] = joblib.load('linear_regression.joblib')
    if os.path.exists('ridge_model.joblib'):
        models['Ridge Regression'] = joblib.load('ridge_model.joblib')
        
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    models = {}

# Load Dataset (Sample)
print("Loading dataset sample...")
try:
    dataset_df = pd.read_csv('dataset/sample_train.tsv', sep='\t')
    # Fill NaN values
    dataset_df['category_name'].fillna('Missing', inplace=True)
    dataset_df['brand_name'].fillna('Missing', inplace=True)
    dataset_df['item_description'].fillna('No description yet', inplace=True)
    
    # Get top categories for filtering
    top_categories = dataset_df['category_name'].value_counts().head(20).index.tolist()
    top_categories.sort()
    
    print("Dataset sample loaded.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset_df = pd.DataFrame()
    top_categories = []

def get_inr_rate():
    """
    Fetches the latest USD to INR exchange rate.
    Returns: rate (float)
    """
    try:
        response = requests.get('https://open.er-api.com/v6/latest/USD')
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return 83.0 # Fallback rate

def get_market_multiplier(category_name, season):
    """
    Simulates market factors based on category and season.
    Returns: (multiplier, reason)
    """
    multiplier = 1.0
    reason = "Standard Market Rate"
    
    # 1. Seasonality Effect
    if season == 'Winter':
        if any(x in category_name.lower() for x in ['coat', 'jacket', 'sweater', 'boots']):
            multiplier *= 1.25
            reason = "High Seasonal Demand (Winter)"
    elif season == 'Summer':
        if any(x in category_name.lower() for x in ['shorts', 't-shirt', 'sandal', 'swim']):
            multiplier *= 1.20
            reason = "High Seasonal Demand (Summer)"
            
    # 2. Category Trends (Mock)
    if 'electronics' in category_name.lower() or 'console' in category_name.lower():
        multiplier *= 1.10
        reason = "Trending Tech Item"
        
    # 3. Brand Premium (Mock)
    if 'apple' in category_name.lower() or 'nike' in category_name.lower():
        multiplier *= 1.15
        reason += " + Brand Premium"
        
    return multiplier, reason

def calculate_optimization_metrics(base_price):
    """
    Generates data for Price vs Profit vs Satisfaction graph.
    """
    prices = []
    profits = []
    satisfactions = []
    
    # Generate price points from -30% to +30% of base price
    for i in range(-30, 31, 5):
        factor = 1 + (i / 100)
        price_point = base_price * factor
        
        # Mock Profit: Assuming cost is 70% of base_price (fixed cost)
        # Profit = Selling Price - Cost
        cost = base_price * 0.70
        profit = price_point - cost
        
        # Mock Satisfaction: 100% at -30% price, 0% at +30% price
        # Linear decay
        satisfaction = 100 - ((i + 30) * (100 / 60))
        satisfaction = max(0, min(100, satisfaction))
        
        prices.append(round(price_point, 2))
        profits.append(round(profit, 2))
        satisfactions.append(round(satisfaction, 1))
        
    # Define Thresholds (e.g., +/- 10% of base price)
    lower_threshold = round(base_price * 0.90, 2)
    upper_threshold = round(base_price * 1.10, 2)
        
    return {
        'prices': prices,
        'profits': profits,
        'satisfactions': satisfactions,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold
    }

def get_market_distribution(category_name):
    """
    Returns a sample of prices for the given category for the histogram.
    """
    try:
        if dataset_df.empty:
            return []
        
        # Filter by category
        category_prices = dataset_df[dataset_df['category_name'] == category_name]['price'].tolist()
        
        # If not enough data, take a random sample from overall dataset
        if len(category_prices) < 10:
             category_prices = dataset_df['price'].sample(n=min(100, len(dataset_df))).tolist()
        
        # Cap at 100 items for performance and cleaner chart
        if len(category_prices) > 100:
            import random
            category_prices = random.sample(category_prices, 100)
            
        return category_prices
    except Exception as e:
        print(f"Error getting market distribution: {e}")
        return []

def get_seasonality_comparison(category_name, base_price):
    """
    Returns prices for different seasons.
    """
    seasons = ['Normal', 'Winter', 'Summer']
    data = []
    
    for season in seasons:
        multiplier, _ = get_market_multiplier(category_name, season)
        price = base_price * multiplier
        data.append({
            'season': season,
            'price': round(price, 2),
            'multiplier': multiplier
        })
        
    return data

def get_price_components(base_price, category_name, season):
    """
    Breaks down the price into components.
    """
    multiplier, reason = get_market_multiplier(category_name, season)
    
    # Components
    base_component = base_price
    market_component = (base_price * multiplier) - base_price
    
    # If negative market component (discount), handle separately or just show net
    # For simplicity, we'll show Base vs Market Adjustment
    
    return {
        'base_price': round(base_component, 2),
        'market_adjustment': round(market_component, 2),
        'reason': reason
    }

def preprocess_input(data):
    # Create DataFrame from input dict if it's not already a DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
    
    # Transform features
    X_name = cv_name.transform(df['name'])
    X_category = cv_category.transform(df['category_name'])
    X_description = tv_description.transform(df['item_description'])
    X_brand = lb_brand.transform(df['brand_name'])
    X_dummies = ohe_condition.transform(df[['item_condition_id', 'shipping']])
    
    # Stack features (Order must match training!)
    # Order from notebook: X_dummies, X_description, X_brand, X_category, X_name
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    
    return sparse_merge

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from form
            data = {
                'name': request.form['name'],
                'brand_name': request.form['brand_name'],
                'category_name': request.form['category_name'],
                'item_condition_id': int(request.form['item_condition_id']),
                'shipping': int(request.form['shipping']),
                'item_description': request.form['item_description']
            }
            
            # Preprocess
            X_input = preprocess_input(data)

            # Predict - Ridge Regression Only
            model_ridge = models.get('Ridge Regression')
            
            if model_ridge:
                log_price = model_ridge.predict(X_input)
            else:
                raise Exception("Ridge Regression model not found")

            base_price = np.expm1(log_price)[0] # Reverse log1p
            
            # Optimization Metrics
            optimization_data = calculate_optimization_metrics(base_price)
            
            # Dynamic Pricing Simulation
            season = request.form.get('season', 'Normal')
            market_multiplier, reason = get_market_multiplier(data['category_name'], season)
            final_price = base_price * market_multiplier
            
            # New Visualizations Data
            market_distribution = get_market_distribution(data['category_name'])
            seasonality_data = get_seasonality_comparison(data['category_name'], base_price)
            price_components = get_price_components(base_price, data['category_name'], season)
            
            # Currency Conversion
            inr_rate = get_inr_rate()
            final_price_inr = final_price * inr_rate
            
            return render_template('index.html', 
                                   models=models.keys(), 
                                   base_price=f"${base_price:.2f}",
                                   market_adjustment=f"{reason} (x{market_multiplier:.2f})",
                                   prediction=f"${final_price:.2f}", 
                                   prediction_inr=f"₹{final_price_inr:.2f}",
                                   input_data=data,
                                   selected_season=season,
                                   optimization_data=optimization_data,
                                   market_distribution=market_distribution,
                                   seasonality_data=seasonality_data,
                                   price_components=price_components,
                                   exchange_rate=inr_rate)
                                   
        except Exception as e:
            return render_template('index.html', models=models.keys(), error=f"Error: {str(e)}")

@app.route('/inventory', methods=['GET'])
def inventory():
    page = request.args.get('page', 1, type=int)
    season = request.args.get('season', 'Normal')
    search_query = request.args.get('search', '').strip()
    selected_category = request.args.get('category', '')
    per_page = 20
    
    if dataset_df.empty:
        return render_template('inventory.html', products=[], page=1, selected_season=season, categories=[], search_query='', selected_category='')

    # Filter dataset
    filtered_df = dataset_df.copy()
    
    if selected_category:
        filtered_df = filtered_df[filtered_df['category_name'] == selected_category]
        
    if search_query:
        filtered_df = filtered_df[
            filtered_df['name'].str.contains(search_query, case=False, na=False) |
            filtered_df['brand_name'].str.contains(search_query, case=False, na=False)
        ]

    if filtered_df.empty:
        return render_template('inventory.html', products=[], page=1, selected_season=season, categories=top_categories, search_query=search_query, selected_category=selected_category)

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    
    # Get chunk
    chunk = filtered_df.iloc[start:end].copy()
    
    if chunk.empty and page > 1:
        # If page is out of range, redirect to page 1 or show empty
        return render_template('inventory.html', products=[], page=page, selected_season=season, categories=top_categories, search_query=search_query, selected_category=selected_category)

    # Preprocess
    X_input = preprocess_input(chunk)
    
    # Predict (using Ridge Model)
    model_ridge = models.get('Ridge Regression')
    
    if model_ridge:
        log_prices = model_ridge.predict(X_input)
    else:
        log_prices = np.zeros(chunk.shape[0]) # Fallback
    
    # log_prices is already calculated above
    base_prices = np.expm1(log_prices)
    
    inr_rate = get_inr_rate()
    
    products = []
    for i, (idx, row) in enumerate(chunk.iterrows()):
        base_price = base_prices[i]
        multiplier, reason = get_market_multiplier(row['category_name'], season)
        dynamic_price = base_price * multiplier
        dynamic_price_inr = dynamic_price * inr_rate
        
        products.append({
            'name': row['name'],
            'category_name': row['category_name'],
            'brand_name': row['brand_name'],
            'item_condition_id': row['item_condition_id'],
            'item_description': row['item_description'],
            'base_price': base_price,
            'dynamic_price': dynamic_price,
            'dynamic_price_inr': dynamic_price_inr,
            'multiplier': multiplier,
            'reason': reason
        })
        
    return render_template('inventory.html', 
                           products=products, 
                           page=page, 
                           selected_season=season, 
                           categories=top_categories, 
                           search_query=search_query, 
                           selected_category=selected_category)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
