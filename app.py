from flask import Flask, render_template, request
import joblib
import numpy as np
from dashboard import add_dashboard  # Import the add_dashboard function

app = Flask(__name__)

# Load the pre-trained Decision Tree Regressor model
model = joblib.load('decision_tree_regressor_model.pkl')

# Integrate Dash into Flask
add_dashboard(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get form inputs with error handling in case fields are missing or invalid
        price = request.form.get('price')
        number_sold = request.form.get('number_sold')
        stock_levels = request.form.get('stock_levels')
        lead_times = request.form.get('lead_times')

        # Check if all fields are filled out, and handle missing input
        if not price or not number_sold or not stock_levels or not lead_times:
            return render_template('result.html', result="Error: Please fill in all fields.")

        # Convert input values to correct types
        price = float(price)
        number_sold = int(number_sold)
        stock_levels = int(stock_levels)
        lead_times = int(lead_times)

        # Prepare input for the model (ensure all 4 features are included)
        input_data = np.array([[price, number_sold, stock_levels, lead_times]])

        # Predict revenue using the Decision Tree Regressor
        predicted_revenue = model.predict(input_data)[0]

        # Return the result to the user
        return render_template('result.html', result=f"Predicted Revenue: ${predicted_revenue:.2f}")

    except Exception as e:
        # Handle any errors gracefully
        return render_template('result.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5005)  # Change port if needed
