import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset into a pandas DataFrame
df = pd.read_csv('supply_chain_data.csv')  # Replace with the actual path to your data

# Print the columns of the DataFrame to debug
print("Columns in the dataset:", df.columns)

# Initialize LabelEncoder
encoder = LabelEncoder()

# Check if the 'Product type' column exists before proceeding
if 'Product type' in df.columns:
    # Encode the 'Product type' column
    df['product_type_encoded'] = encoder.fit_transform(df['Product type'])

    # Save the encoder to a file
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print("Label encoder saved successfully.")
else:
    print("Error: 'Product type' column not found in the dataset.")
