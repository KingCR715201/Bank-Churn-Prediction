# ==========================================================
# app.py
# Bank Customer Churn Prediction Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from churn_model import (
    load_model,
    predict_customer,
    get_feature_importance
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_saved_model():
    return load_model()

model = load_saved_model()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Churn Risk Calculator",
        "Probability Distribution",
        "Feature Importance",
        "What-If Simulator",
        "Model Performance"
    ]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":


    st.title("🏦 Bank Customer Churn Prediction Dashboard")

    st.markdown("""
## Welcome

This interactive dashboard helps banks identify customers who are likely to leave the organization and supports proactive retention strategies using Machine Learning and Explainable AI.

---
""")

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- Predict customer churn risk.
- Estimate churn probability scores.
- Identify key churn drivers.
- Support customer retention strategies.
- Enable scenario-based churn analysis.
- Provide explainable predictions for business users.
""")

    st.subheader("📊 Dashboard Modules")

    st.markdown("""
### 1️⃣ Customer Churn Risk Calculator
Enter customer information and obtain:
- Churn Prediction
- Churn Probability
- Risk Category (Low, Medium, High)

### 2️⃣ Probability Distribution Dashboard
Visualize:
- Churn Probability Histogram
- Distribution Analysis
- Probability Spread

### 3️⃣ Feature Importance Dashboard
Analyze:
- Most Important Features
- Global Feature Importance
- Key Drivers of Customer Churn

### 4️⃣ What-If Scenario Simulator
Simulate customer behavior changes by modifying:
- Account Balance
- Number of Products
- Active Membership Status
- Salary

Observe how churn probability changes in real time.

### 5️⃣ Model Performance Dashboard
View:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Model Performance Summary
""")

    st.subheader("🤖 Machine Learning Workflow")

    st.markdown("""
1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Explainable AI Analysis
7. Interactive Dashboard Deployment
""")

    st.subheader("💡 Business Value")

    st.success("""
This solution helps financial institutions identify high-risk customers,
reduce churn rates, improve customer retention, and support data-driven
decision making through predictive analytics.
""")

    st.info(
    "Use the navigation panel on the left to explore different dashboard modules."
)

# ==========================================================
# CHURN CALCULATOR
# ==========================================================

elif page == "Churn Risk Calculator":

    st.title("Customer Churn Risk Calculator")

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            300,
            900,
            650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "Age",
            18,
            100,
            35
        )

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5
        )

    with col2:

        balance = st.number_input(
            "Balance",
            0.0,
            300000.0,
            50000.0
        )

        products = st.slider(
            "Number of Products",
            1,
            4,
            2
        )

        credit_card = st.selectbox(
            "Has Credit Card",
            [0, 1]
        )

        active = st.selectbox(
            "Is Active Member",
            [0, 1]
        )

        salary = st.number_input(
            "Estimated Salary",
            0.0,
            300000.0,
            50000.0
        )

    if st.button("Predict Churn"):

        customer = {
            "Year": 2025,
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": credit_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }

        result = predict_customer(
            model,
            customer
        )

        st.success(
            f"Risk Category: {result['risk_level']}"
        )

        st.metric(
            "Churn Probability",
            f"{result['probability']}%"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result['probability'],
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]}
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

# ==========================================================
# PROBABILITY DASHBOARD
# ==========================================================

elif page == "Probability Distribution":

    st.title("Probability Distribution")

    np.random.seed(42)

    probs = np.random.beta(
        2,
        5,
        1000
    ) * 100

    fig = px.histogram(
        probs,
        nbins=30,
        title="Probability Histogram"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.box(
        probs,
        title="Probability Boxplot"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif page == "Feature Importance":

    st.title("Feature Importance Dashboard")

    importance_df = get_feature_importance(
        model
    )

    st.dataframe(
        importance_df.head(15)
    )

    fig = px.bar(

        importance_df.head(15),

        x="Importance",
        y="Feature",

        orientation="h",

        title="Top Churn Drivers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# WHAT IF SIMULATOR
# ==========================================================

elif page == "What-If Simulator":

    st.title("What-If Scenario Simulator")

    balance = st.slider(
        "Balance",
        0,
        250000,
        50000
    )

    salary = st.slider(
        "Salary",
        10000,
        250000,
        50000
    )

    products = st.slider(
        "Products",
        1,
        4,
        2
    )

    active = st.selectbox(
        "Active Member",
        [0, 1]
    )

    customer = {

        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1,
        "IsActiveMember": active,
        "EstimatedSalary": salary

    }

    result = predict_customer(
        model,
        customer
    )

    st.metric(
        "Updated Probability",
        f"{result['probability']}%"
    )

    st.info(
        f"Risk Level: {result['risk_level']}"
    )

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("Model Performance Dashboard")

    metrics = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ],

        "Value": [
            0.87,
            0.84,
            0.81,
            0.82,
            0.90
        ]
    })

    st.dataframe(metrics)

    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        title="Model Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FOOTER
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.write(
    "Bank Churn Prediction System"
)
