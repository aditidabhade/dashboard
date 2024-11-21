import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving the model

# Load your dataset (make sure your dataset path is correct)
# For example, replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('supply_chain_data.csv')

# Check the columns of the dataset
print(f"Columns in the dataset: {data.columns}")

# Assume 'Revenue generated' is the target variable and the rest are features
target = 'Revenue generated'

# Define the feature columns (excluding target variable)
# You can select columns as per your dataset
feature_columns = ['Price', 'Number of products sold', 'Stock levels', 'Lead times']  # Adjust this based on your dataset

# Ensure that you have no missing values in the selected columns (drop or fill NaNs)
data = data.dropna(subset=feature_columns + [target])

# Split data into features (X) and target (y)
X = data[feature_columns]
y = data[target]

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
print(f"Mean Squared Error: {mse}")

# Print the predicted vs actual values for the test set
print("\nPredictions vs Actuals:")
for actual, predicted in zip(y_test.head(), y_pred[:5]):
    print(f"Actual: {actual}, Predicted: {predicted}")

# Save the trained model using joblib
model_filename = 'decision_tree_regressor_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

# Example of predicting for new data (input_data should match the features used in training)
input_data = np.array([[200, 150, 20, 30]])  # Example new data (same number of features as training)
predicted_revenue = model.predict(input_data)[0]
print(f"\nPredicted Revenue for the input data: {predicted_revenue}")

# Load the saved model (to use it later)
loaded_model = joblib.load(model_filename)

# Predict using the loaded model
loaded_model_prediction = loaded_model.predict(input_data)[0]
print(f"\nPredicted Revenue using loaded model: {loaded_model_prediction}")
