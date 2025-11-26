Simple Churn Prediction Application

This is a Python script that demonstrates a basic machine learning workflow for predicting customer churn using a synthetic dataset containing mock Personally Identifiable Information (PII).

The script performs data loading, essential preprocessing (dropping PII, scaling numerical features), model training (Logistic Regression), evaluation, and sample prediction.

Prerequisites

To run this application, you must have the following Python libraries installed:

pip install pandas scikit-learn numpy


You also need the following two files in the same directory:

ml_predictor.py (The main Python script)

fake_ml_data.csv (The generated synthetic dataset with 200 entries)

How to Run

Execute the Python script directly from your terminal:

python ml_predictor.py


Workflow Overview

Data Loading: Reads fake_ml_data.csv.

PII Handling: Drops columns identified as PII (User_ID, Full_Name, Email, Street_Address).

Preprocessing: Scales numerical features (Age, Monthly_Usage_Hours, Support_Tickets) using StandardScaler.

Training: Trains a LogisticRegression model on 80% of the data.

Evaluation: Reports the model's accuracy and performance metrics on the remaining 20% test set.

Prediction: Demonstrates predictions for two example customer profiles: one low-risk and one high-risk.

Sample Output

The output of the script will look similar to the following, though the exact accuracy and probability scores may vary slightly based on the random split:

Data loaded successfully. Total entries: 200

Features after preprocessing and scaling:
   Age  Monthly_Usage_Hours  Support_Tickets  Is_Premium
0 -0.472856            -0.108930         0.207904           1
1  0.963471            -0.965768        -0.547141           0
2 -1.416172             1.884144        -0.924663           1
3  0.024476            -1.468817         1.717992           0
4  1.450799            -0.370422        -0.169618           1

Training set size: 160 samples
Testing set size: 40 samples

==================================================
Model Training Complete (Logistic Regression)
==================================================
Accuracy on Test Set: 0.8250

Classification Report (0=Not Churned, 1=Churned):
               precision    recall  f1-score   support

           0       0.78      0.84      0.81        20
           1       0.86      0.81      0.83        20

    accuracy                           0.82        40
   macro avg       0.82      0.82      0.82        40
weighted avg       0.82      0.82      0.82        40

==================================================

--- Testing Prediction on Custom Profiles ---

--- New Customer Profile ---
Age: 35, Usage: 250.0 hrs, Tickets: 0, Premium: Yes
Predicted Status: NOT CHURN
Probability (Not Churn / Churn): 0.95 / 0.05

--- New Customer Profile ---
Age: 55, Usage: 15.0 hrs, Tickets: 8, Premium: No
Predicted Status: CHURN
Probability (Not Churn / Churn): 0.10 / 0.90
