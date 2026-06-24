# Bank Customer Churn Prediction System

## Overview

This project predicts whether a bank customer is likely to churn using Machine Learning and Explainable AI techniques.

The application includes:

* Customer Churn Prediction
* Churn Probability Estimation
* Risk Categorization
* Feature Importance Analysis
* What-If Scenario Simulation
* Interactive Streamlit Dashboard

---

## Dataset Features

### Input Features

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

### Target Variable

* Exited

---

## Feature Engineering

The following features are generated:

### Balance_to_Salary_Ratio

Balance / (EstimatedSalary + 1)

### Products_Per_Tenure

NumOfProducts / (Tenure + 1)

### Age_Tenure_Interaction

Age × Tenure

### Customer_Engagement_Score

HasCrCard + IsActiveMember + NumOfProducts

---

## Machine Learning Workflow

1. Data Cleaning
2. Missing Value Handling
3. One-Hot Encoding
4. Feature Scaling
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Explainable AI
9. Deployment

---

## Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost (Optional)

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Streamlit Dashboard

### Home

Project Overview and Objectives

### Churn Risk Calculator

Predict customer churn probability.

### Probability Dashboard

Distribution Analysis

### Feature Importance

Top churn drivers.

### What-If Simulator

Analyze probability changes under different scenarios.

### Model Performance

Model metrics and performance comparison.

---

## Running the Project

### Train Model

python churn_model.py

### Launch Dashboard

streamlit run app.py

---

## Project Structure

Bank_Churn_Prediction/

├── churn_model.py

├── app.py

├── requirements.txt

├── README.md

└── DEPLOYMENT.md

---

## Business Benefits

* Customer Retention
* Revenue Protection
* Risk Assessment
* Targeted Marketing
* Customer Segmentation

---

## Future Enhancements

* Batch Prediction
* Cloud Deployment
* Real-Time Prediction API
* Advanced SHAP Visualizations
* Deep Learning Models
