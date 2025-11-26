import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. Load Data ---
# Note: This script assumes 'fake_ml_data.csv' is in the same directory.
try:
    df = pd.read_csv('fake_ml_data.csv')
    print("Data loaded successfully. Total entries:", len(df))
except FileNotFoundError:
    print("Error: 'fake_ml_data.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---

# Define columns to drop (PII and unique identifiers are irrelevant for training)
pii_columns = ['User_ID', 'Full_Name', 'Email', 'Street_Address']
df_clean = df.drop(columns=pii_columns)

# Convert boolean columns to integer (0 or 1)
df_clean['Is_Premium'] = df_clean['Is_Premium'].astype(int)
df_clean['Churned'] = df_clean['Churned'].astype(int)

# Define features (X) and target (y)
X = df_clean.drop('Churned', axis=1)
y = df_clean['Churned']

# Identify numerical features for scaling
numerical_features = ['Age', 'Monthly_Usage_Hours', 'Support_Tickets']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nFeatures after preprocessing and scaling:")
print(X.head())

# --- 3. Model Training ---

# Split the data into training and testing sets
# We use a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Initialize and train the Logistic Regression model
# Logistic Regression is a good baseline model for binary classification (like Churn)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n" + "="*50)
print("Model Training Complete (Logistic Regression)")
print("="*50)
print(f"Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report (0=Not Churned, 1=Churned):\n", report)
print("="*50)

# --- 5. Prediction Example ---

# Let's create a new customer profile to predict churn risk
# Example 1: High usage, low support tickets (Low Churn Risk)
new_customer_data_low_risk = pd.DataFrame({
    'Age': [35],
    'Monthly_Usage_Hours': [250.0],  # High usage
    'Support_Tickets': [0],          # Low tickets
    'Is_Premium': [1]                # Premium
})

# Example 2: Low usage, high support tickets (High Churn Risk)
new_customer_data_high_risk = pd.DataFrame({
    'Age': [55],
    'Monthly_Usage_Hours': [15.0],   # Low usage
    'Support_Tickets': [8],          # High tickets
    'Is_Premium': [0]                # Not Premium
})

# Remember to scale the numerical features of the new data using the SAME scaler object
# The order of columns must match the training data
X_new_low = new_customer_data_low_risk.copy()
X_new_low[numerical_features] = scaler.transform(X_new_low[numerical_features])
X_new_low = X_new_low[X_train.columns] # Reorder columns

X_new_high = new_customer_data_high_risk.copy()
X_new_high[numerical_features] = scaler.transform(X_new_high[numerical_features])
X_new_high = X_new_high[X_train.columns] # Reorder columns

# Make predictions
pred_low = model.predict(X_new_low)
proba_low = model.predict_proba(X_new_low)[0]

pred_high = model.predict(X_new_high)
proba_high = model.predict_proba(X_new_high)[0]

def print_prediction(data, pred, proba):
    """Helper function to print prediction results clearly."""
    status = "CHURN" if pred[0] == 1 else "NOT CHURN"
    print(f"\n--- New Customer Profile ---")
    print(f"Age: {data['Age'].iloc[0]}, Usage: {data['Monthly_Usage_Hours'].iloc[0]} hrs, Tickets: {data['Support_Tickets'].iloc[0]}, Premium: {'Yes' if data['Is_Premium'].iloc[0] == 1 else 'No'}")
    print(f"Predicted Status: {status}")
    print(f"Probability (Not Churn / Churn): {proba[0]:.2f} / {proba[1]:.2f}")

print("\n--- Testing Prediction on Custom Profiles ---")
print_prediction(new_customer_data_low_risk, pred_low, proba_low)
print_prediction(new_customer_data_high_risk, pred_high, proba_high)
