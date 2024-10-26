from flask import Flask, request, render_template
import pickle
import numpy as np
from dashboard import add_dashboard  # Import the dashboard integration function

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Attach the dashboard to the Flask app
dash_app = add_dashboard(app)

@app.route('/')
def home():
    return render_template("index.html")

from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    try:
        product_type = request.form['product_type']
        sku = request.form['sku']
        price = float(request.form['price'])
        availability = float(request.form['availability'])
        number_sold = int(request.form['number_sold'])
        revenue_generated = float(request.form['revenue_generated'])
        customer_demographics = request.form['customer_demographics']
        stock_levels = float(request.form['stock_levels'])
        lead_times = float(request.form['lead_times'])
        order_quantities = float(request.form['order_quantities'])
        shipping_times = float(request.form['shipping_times'])
        shipping_carriers = request.form['shipping_carriers']
        shipping_costs = float(request.form['shipping_costs'])
        supplier_name = request.form['supplier_name']
        location = request.form['location']
        manufacturing_lead_time = float(request.form['manufacturing_lead_time'])
        inspection_results = request.form['inspection_results']
        costs = float(request.form['costs'])
        defect_rates = float(request.form['defect_rates'])
        transportation_modes = request.form['transportation_modes']
        routes = request.form['routes']
        production_volumes = float(request.form['production_volumes'])  # Ensure this is here
    except KeyError as e:
        return f"Missing data for {e}", 400  # Return an error if any field is missing

# Convert categorical data to appropriate numerical encoding if necessary
    category_mapping = {
        'haircare': 0,
        'skincare': 1,
        'makeup': 2,
        'fragrance': 3  # Extend this as per your requirements
    }

    product_type_encoded = category_mapping.get(product_type.lower(), -1)
    if product_type_encoded == -1:
        return render_template('index.html', pred="Error: Invalid product type.")

    # Prepare features for prediction, ensure to include all expected features
    features = np.array([[product_type_encoded, sku, price, availability, number_sold, revenue_generated,
                          customer_demographics, stock_levels, lead_times, order_quantities, shipping_times,
                          shipping_carriers, shipping_costs, supplier_name, location,
                          manufacturing_lead_time, production_volumes, 
                          inspection_results, defect_rates, transportation_modes, routes, costs]])

    # Make prediction using the pre-trained model
    predicted_cost = model.predict(features)

    # Format the predicted cost to two decimal places
    output = f"{predicted_cost[0]:.2f}"

    return render_template('index.html', pred=f'Predicted Cost: {output}',
                           product_type=product_type, availability=availability, 
                           number_sold=number_sold, stock_levels=stock_levels, 
                           lead_times=lead_times, production_volumes=production_volumes,
                           defect_rates=defect_rates) 
    
if __name__ == '__main__':
    dash_app.run_server(debug=True,port=8050)
