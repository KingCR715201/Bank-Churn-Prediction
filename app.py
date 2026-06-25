import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🏦 Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main{
    padding-top:10px;
}

div[data-testid="metric-container"]{
    background:#f8f9fa;
    border-radius:12px;
    padding:18px;
    border-left:5px solid #1f77b4;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/churn_model.pkl")

    X_test = joblib.load("models/X_test.pkl")

    y_test = joblib.load("models/y_test.pkl")

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

    model_name = joblib.load(
        "models/model_name.pkl"
    )

    with open("models/metrics.json") as f:

        metrics = json.load(f)

    return (
        model,
        X_test,
        y_test,
        feature_names,
        model_name,
        metrics,
    )


(
    model,
    X_test,
    y_test,
    feature_names,
    model_name,
    metrics,
) = load_artifacts()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv("European_Bank.csv")


df = load_dataset()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",

        "📊 Dashboard",

        "🎯 Predict Churn",

        "📈 Probability Analysis",

        "💡 Explainable AI",

        "⚙ What-if Simulator",

        "📉 Model Performance",

    ]

)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🏦 Bank Customer Churn Prediction")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df),
    )

    c2.metric(
        "Features",
        len(df.columns) - 1,
    )

    c3.metric(
        "Best Model",
        model_name,
    )

    c4.metric(
        "ROC-AUC",
        f"{metrics['ROC AUC']:.3f}",
    )

    st.markdown("---")

    st.header("Project Overview")

    st.write(
        """
This project predicts whether a customer will leave the bank using Machine Learning.

### Objectives

- Predict customer churn
- Estimate churn probability
- Explain predictions using SHAP
- Analyze important churn drivers
- Support customer retention strategies

"""
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

# ============================================================
# DASHBOARD PAGE
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Customer Churn Dashboard")

    churn_rate = df["Exited"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df),
    )

    c2.metric(
        "Churn Rate",
        f"{churn_rate:.2f} %",
    )

    c3.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.3f}",
    )

    c4.metric(
        "ROC-AUC",
        f"{metrics['ROC AUC']:.3f}",
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        fig = px.histogram(

            df,

            x="Age",

            color="Exited",

            title="Age Distribution",

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        fig = px.box(

            df,

            x="Exited",

            y="Balance",

            title="Balance vs Churn",

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        fig = px.pie(

            df,

            names="Geography",

            title="Customer Geography",

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        fig = px.bar(

            df.groupby("Gender")["Exited"]

            .mean()

            .reset_index(),

            x="Gender",

            y="Exited",

            title="Gender-wise Churn Rate",

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

# ============================================================
# PREDICT CHURN PAGE
# ============================================================

elif page == "🎯 Predict Churn":

    st.title("🎯 Customer Churn Prediction")

    st.write("Enter customer information to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
        )

        geography = st.selectbox(
            "Geography",
            sorted(df["Geography"].unique()),
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].unique()),
        )

        age = st.slider(
            "Age",
            18,
            100,
            35,
        )

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5,
        )

    with col2:

        balance = st.number_input(
            "Balance",
            value=50000.0,
        )

        products = st.slider(
            "Number of Products",
            1,
            4,
            2,
        )

        card = st.selectbox(
            "Has Credit Card",
            [0, 1],
        )

        active = st.selectbox(
            "Is Active Member",
            [0, 1],
        )

        salary = st.number_input(
            "Estimated Salary",
            value=75000.0,
        )

    # ============================================================
# CREATE CUSTOMER DATA
# ============================================================

customer = pd.DataFrame(index=[0])

customer["CreditScore"] = credit_score
customer["Geography"] = geography
customer["Gender"] = gender
customer["Age"] = age
customer["Tenure"] = tenure
customer["Balance"] = balance
customer["NumOfProducts"] = products
customer["HasCrCard"] = card
customer["IsActiveMember"] = active
customer["EstimatedSalary"] = salary

# ============================================================
# FEATURE ENGINEERING
# ============================================================

customer["Balance_to_Salary_Ratio"] = (
    customer["Balance"] /
    (customer["EstimatedSalary"] + 1)
)

customer["Products_Per_Tenure"] = (
    customer["NumOfProducts"] /
    (customer["Tenure"] + 1)
)

customer["Age_Tenure_Interaction"] = (
    customer["Age"] *
    customer["Tenure"]
)

customer["Customer_Engagement_Score"] = (
    customer["IsActiveMember"] * 2
    + customer["HasCrCard"]
    + customer["NumOfProducts"]
)

# ============================================================
# MATCH TRAINING COLUMNS
# ============================================================

expected_columns = list(
    model.named_steps["preprocessor"].feature_names_in_
)

# Add any missing columns
for col in expected_columns:
    if col not in customer.columns:
        customer[col] = 0

# Remove extra columns
customer = customer[expected_columns]

# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Customer Churn"):

    probability = float(
        model.predict_proba(customer)[0][1]
    )

    prediction = int(
        model.predict(customer)[0]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        if probability < 0.30:
            st.success("🟢 Low Risk")
        elif probability < 0.70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

    with col2:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"},
                    ],
                },
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Business Recommendation")

        if probability < 0.30:

            st.success("""
Customer has a low churn risk.

Recommendation:
- Continue regular engagement
- Offer loyalty rewards
""")

        elif probability < 0.70:

            st.warning("""
Customer has moderate churn risk.

Recommendation:
- Offer personalized promotions
- Increase customer engagement
""")

        else:

            st.error("""
Customer has high churn risk.

Recommendation:
- Immediate relationship manager follow-up
- Premium retention offer
- Personalized banking services
""")

# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

elif page == "📈 Probability Analysis":

    st.title("📈 Churn Probability Distribution")

    probabilities = model.predict_proba(X_test)[:,1]

    tab1, tab2, tab3 = st.tabs([

        "Histogram",

        "Density",

        "Box Plot"

    ])

    with tab1:

        fig = px.histogram(

            x=probabilities,

            nbins=30,

            title="Probability Histogram"

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab2:

        fig = px.violin(

            y=probabilities,

            box=True,

            points="all",

            title="Probability Density"

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab3:

        fig = px.box(

            y=probabilities,

            title="Probability Box Plot"

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.metric(

        "Average Churn Probability",

        f"{probabilities.mean():.2%}"

    )

# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "💡 Explainable AI":

    st.title("💡 Explainable AI")

    classifier = model.named_steps["classifier"]

    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    st.subheader("Top Feature Importance")

    if hasattr(classifier, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": classifier.feature_importances_

        }).sort_values(

            "Importance",

            ascending=False

        )

    elif hasattr(classifier, "coef_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": np.abs(classifier.coef_[0])

        }).sort_values(

            "Importance",

            ascending=False

        )

    else:

        importance = None

    if importance is not None:

        fig = px.bar(

            importance.head(20),

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            title="Top 20 Features"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("SHAP Summary Plot")

    try:

        import shap

        if hasattr(classifier, "feature_importances_"):

            explainer = shap.TreeExplainer(classifier)

            shap_values = explainer.shap_values(transformed)

            fig, ax = plt.subplots(figsize=(10,6))

            shap.summary_plot(

                shap_values,

                transformed,

                feature_names=feature_names,

                show=False

            )

            st.pyplot(fig)

        else:

            st.info(
                "SHAP Summary is available for tree-based models."
            )

    except Exception as e:

        st.error(e)

# ============================================================
# WHAT IF SIMULATOR
# ============================================================

elif page == "⚙ What-if Simulator":

    st.title("⚙ What-if Scenario Simulator")

    sample = X_test.iloc[[0]].copy()

    balance = st.slider(

        "Balance",

        0,

        250000,

        int(sample["Balance"].iloc[0]),

        step=1000

    )

    salary = st.slider(

        "Salary",

        0,

        250000,

        int(sample["EstimatedSalary"].iloc[0]),

        step=1000

    )

    products = st.slider(

        "Products",

        1,

        4,

        int(sample["NumOfProducts"].iloc[0])

    )

    active = st.selectbox(

        "Active Member",

        [0,1],

        index=int(sample["IsActiveMember"].iloc[0])

    )

    sample["Balance"] = balance

    sample["EstimatedSalary"] = salary

    sample["NumOfProducts"] = products

    sample["IsActiveMember"] = active

    sample["Balance_to_Salary_Ratio"] = (

        sample["Balance"] /

        (sample["EstimatedSalary"]+1)

    )

    sample["Products_Per_Tenure"] = (

        sample["NumOfProducts"] /

        (sample["Tenure"]+1)

    )

    sample["Age_Tenure_Interaction"] = (

        sample["Age"] *

        sample["Tenure"]

    )

    sample["Customer_Engagement_Score"] = (

        sample["IsActiveMember"]*2

        +

        sample["HasCrCard"]

        +

        sample["NumOfProducts"]

    )

    probability = model.predict_proba(

        sample

    )[0][1]

    st.metric(

        "Updated Churn Probability",

        f"{probability:.2%}"

    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📉 Model Performance":

    st.title("📉 Model Performance")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(

        "Accuracy",

        f"{metrics['Accuracy']:.3f}"

    )

    c2.metric(

        "Precision",

        f"{metrics['Precision']:.3f}"

    )

    c3.metric(

        "Recall",

        f"{metrics['Recall']:.3f}"

    )

    c4.metric(

        "F1 Score",

        f"{metrics['F1 Score']:.3f}"

    )

    c5.metric(

        "ROC AUC",

        f"{metrics['ROC AUC']:.3f}"

    )

    st.markdown("---")

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:,1]

    cm = confusion_matrix(

        y_test,

        prediction

    )

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        ax=ax

    )

    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.markdown("---")

    fpr,tpr,_ = roc_curve(

        y_test,

        probability

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=fpr,

            y=tpr,

            mode="lines",

            name=f"AUC = {roc_auc:.3f}"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=[0,1],

            y=[0,1],

            mode="lines",

            line=dict(dash="dash"),

            name="Random"

        )

    )

    fig.update_layout(

        title="ROC Curve",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    output = X_test.copy()

    output["Actual"] = y_test

    output["Prediction"] = prediction

    output["Probability"] = probability

    csv = output.to_csv(index=False)

    st.download_button(

        "⬇ Download Predictions",

        csv,

        "predictions.csv",

        "text/csv"

    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(

"""
Bank Customer Churn Prediction Dashboard

Developed using:

• Python

• Scikit-Learn

• Plotly

• Streamlit

• SHAP Explainable AI
"""

)
