import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Load your dataset
df = pd.read_csv('supply_chain_data.csv')

# Categorical columns to encode
categorical_columns = ['Product type', 'Customer demographics', 'Shipping carriers', 'Supplier name',
                       'Location', 'Inspection results', 'Transportation modes', 'Routes']

# Apply LabelEncoder to categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Splitting the dataset into features (X) and target (y)
X = df.drop(['Costs', 'SKU'], axis=1)  # Dropping 'Costs' and 'SKU' as it's an identifier
y = df['Costs']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Initializing and training the RandomForestRegressor
model = RandomForestRegressor(random_state=52)
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)

# Calculating performance metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Displaying the results
print("Random Forest Regressor Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Save the trained model as a pickle file
try:
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as 'model.pkl'")
except Exception as e:
    print(f"Error saving the model: {e}")

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

print("Best Random Forest Regressor Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse_best}")
print(f"R-squared (R2): {r2_best}")
print(f"Mean Absolute Error (MAE): {mae_best}")
