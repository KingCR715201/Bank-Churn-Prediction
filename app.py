import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_curve,
    auc,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

h1,h2,h3{
color:#003366;
}

.metric-container{
background:#ffffff;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.15);
text-align:center;
}

.sidebar .sidebar-content{
background:#0E1117;
}

</style>
""",unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

MODEL_DIR="models"

MODEL_PATH=os.path.join(MODEL_DIR,"churn_model.pkl")
METRICS_PATH=os.path.join(MODEL_DIR,"metrics.json")
MODEL_NAME_PATH=os.path.join(MODEL_DIR,"model_name.pkl")
XTEST_PATH=os.path.join(MODEL_DIR,"X_test.pkl")
YTEST_PATH=os.path.join(MODEL_DIR,"y_test.pkl")

model=joblib.load(MODEL_PATH)
model_name=joblib.load(MODEL_NAME_PATH)

X_test=joblib.load(XTEST_PATH)
y_test=joblib.load(YTEST_PATH)

with open(METRICS_PATH,"r") as f:
    metrics=json.load(f)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🏦 Navigation")

page=st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Dashboard",
        "📈 Model Performance"
    ]
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page=="🏠 Home":

    st.title("🏦 Bank Customer Churn Prediction")

    st.markdown("---")

    st.markdown("""
### 📌 Project Description

This project predicts whether a customer is likely to leave the bank.

Machine Learning models were trained on historical customer information to identify customers at risk of churn.

The project performs:

- Data Cleaning
- Feature Engineering
- Model Training
- Model Evaluation
- Customer Churn Prediction
- Interactive Dashboard

---

### Dataset Features

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member
- Estimated Salary

---

### Target Variable

Exited

- 1 → Customer Left
- 0 → Customer Stayed

---
""")

    col1,col2,col3=st.columns(3)

    with col1:
        st.metric("Best Model",model_name)

    with col2:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.2%}"
        )

    with col3:
        st.metric(
            "ROC-AUC",
            f"{metrics['ROC AUC']:.3f}"
        )

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

elif page == "🔮 Prediction":

    st.title("🔮 Bank Customer Churn Prediction")

    st.markdown("### Enter Customer Details")

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

        num_products = st.selectbox(
            "Number of Products",
            [1,2,3,4]
        )

        has_card = st.selectbox(
            "Has Credit Card",
            [0,1]
        )

        active_member = st.selectbox(
            "Is Active Member",
            [0,1]
        )

        salary = st.number_input(
            "Estimated Salary",
            0.0,
            300000.0,
            100000.0
        )

    st.markdown("---")

    if st.button("Predict Churn", use_container_width=True):

        # Feature Engineering
        balance_salary_ratio = balance / (salary + 1)

        products_per_tenure = num_products / (tenure + 1)

        age_tenure = age * tenure

        engagement_score = (
            active_member * 2
            + has_card
            + num_products
        )

        input_df = pd.DataFrame({

            "CreditScore":[credit_score],
            "Geography":[geography],
            "Gender":[gender],
            "Age":[age],
            "Tenure":[tenure],
            "Balance":[balance],
            "NumOfProducts":[num_products],
            "HasCrCard":[has_card],
            "IsActiveMember":[active_member],
            "EstimatedSalary":[salary],

            "Balance_to_Salary_Ratio":[balance_salary_ratio],
            "Products_Per_Tenure":[products_per_tenure],
            "Age_Tenure_Interaction":[age_tenure],
            "Customer_Engagement_Score":[engagement_score]

        })

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Customer is likely to CHURN")

        else:

            st.success("✅ Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

        st.markdown("### Input Summary")

        st.dataframe(
            input_df,
            use_container_width=True
        )

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Customer Churn Dashboard")

    st.markdown("### Dataset Overview")

    df = X_test.copy()
    df["Exited"] = y_test.values

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers", len(df))

    with col2:
        st.metric("Churned", int(df["Exited"].sum()))

    with col3:
        st.metric("Retained", int((df["Exited"] == 0).sum()))

    with col4:
        st.metric(
            "Churn Rate",
            f"{df['Exited'].mean()*100:.2f}%"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # -----------------------
    # Churn Distribution
    # -----------------------

    with col1:

        fig, ax = plt.subplots(figsize=(5,4))

        sns.countplot(
            data=df,
            x="Exited",
            palette="Set2",
            ax=ax
        )

        ax.set_xticklabels(["Stayed","Churned"])

        ax.set_title("Customer Churn Distribution")

        st.pyplot(fig)

    # -----------------------
    # Gender Distribution
    # -----------------------

    with col2:

        if "Gender" in df.columns:

            fig, ax = plt.subplots(figsize=(5,4))

            sns.countplot(
                data=df,
                x="Gender",
                hue="Exited",
                palette="viridis",
                ax=ax
            )

            ax.set_title("Gender vs Churn")

            st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # -----------------------
    # Geography
    # -----------------------

    with col1:

        if "Geography" in df.columns:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.countplot(
                data=df,
                x="Geography",
                hue="Exited",
                palette="coolwarm",
                ax=ax
            )

            ax.set_title("Geography vs Churn")

            st.pyplot(fig)

    # -----------------------
    # Age Distribution
    # -----------------------

    with col2:

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            data=df,
            x="Age",
            hue="Exited",
            kde=True,
            palette="Set1",
            ax=ax
        )

        ax.set_title("Age Distribution")

        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # -----------------------
    # Balance
    # -----------------------

    with col1:

        fig, ax = plt.subplots(figsize=(6,4))

        sns.boxplot(
            data=df,
            x="Exited",
            y="Balance",
            palette="Set3",
            ax=ax
        )

        ax.set_xticklabels(["Stayed","Churned"])

        ax.set_title("Balance vs Churn")

        st.pyplot(fig)

    # -----------------------
    # Active Member
    # -----------------------

    with col2:

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=df,
            x="IsActiveMember",
            hue="Exited",
            palette="Dark2",
            ax=ax
        )

        ax.set_title("Active Member vs Churn")

        st.pyplot(fig)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.markdown("### Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.2%}"
        )

        st.metric(
            "Precision",
            f"{metrics['Precision']:.2%}"
        )

    with col2:
        st.metric(
            "Recall",
            f"{metrics['Recall']:.2%}"
        )

        st.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.2%}"
        )

    with col3:
        st.metric(
            "ROC AUC",
            f"{metrics['ROC AUC']:.3f}"
        )

        st.metric(
            "Best Model",
            model_name
        )

    st.markdown("---")

    st.subheader("Confusion Matrix")

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6,5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("ROC Curve")

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    ax.plot([0,1],[0,1],"k--")

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.set_title("ROC Curve")

    ax.legend()

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Model Information")

    st.write(f"**Selected Model :** {model_name}")

    st.write(f"**Test Samples :** {len(X_test)}")

    st.write(f"**Number of Features :** {X_test.shape[1]}")

    st.markdown("---")

    st.subheader("Download Evaluation Metrics")

    metrics_json = json.dumps(
        metrics,
        indent=4
    )

    st.download_button(
        label="📥 Download metrics.json",
        data=metrics_json,
        file_name="metrics.json",
        mime="application/json"
    )

