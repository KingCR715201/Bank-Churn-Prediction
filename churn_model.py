# ==========================================================
# churn_model.py
# Bank Customer Churn Prediction System
# ==========================================================

import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier
)

import shap
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "best_model.pkl"
DATA_PATH = "C:/Users/paras/OneDrive/My Projects or codes/Unified Mentor Project 1/European_Bank.csv"

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def feature_engineering(df):

    df["Balance_to_Salary_Ratio"] = (
        df["Balance"] /
        (df["EstimatedSalary"] + 1)
    )

    df["Products_Per_Tenure"] = (
        df["NumOfProducts"] /
        (df["Tenure"] + 1)
    )

    df["Age_Tenure_Interaction"] = (
        df["Age"] *
        df["Tenure"]
    )

    df["Customer_Engagement_Score"] = (
        df["HasCrCard"] +
        df["IsActiveMember"] +
        df["NumOfProducts"]
    )

    return df

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(filepath):

    df = pd.read_csv(filepath)

    columns_to_drop = [
    "CustomerId",
    "Surname",
    "Year"
    ]

    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

        df = feature_engineering(df)
    return df

# ==========================================================
# PREPROCESSOR
# ==========================================================

def build_preprocessor(X):

    categorical_features = [
        "Geography",
        "Gender"
    ]

    numerical_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor

# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(filepath):

    print("Loading Dataset...")

    df = load_data(filepath)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    print("Training Model...")

    pipeline.fit(
        X_train,
        y_train
    )

    evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    save_model(pipeline)

    return pipeline

# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_model(
        model,
        X_test,
        y_test
):

    preds = model.predict(X_test)

    probs = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    auc = roc_auc_score(
        y_test,
        probs
    )

    print("\nMODEL PERFORMANCE")
    print("-" * 40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            preds
        )
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            preds
        )
    )

# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel Saved -> {MODEL_PATH}"
    )

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():
    return joblib.load("best_model.pkl")

# ==========================================================
# RISK CATEGORY
# ==========================================================

def get_risk_level(probability):

    if probability <= 0.30:
        return "Low Risk"

    elif probability <= 0.70:
        return "Medium Risk"

    else:
        return "High Risk"

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_customer(
        model,
        customer_data
):

    df = pd.DataFrame(
        [customer_data]
    )

    df = feature_engineering(df)

    probability = model.predict_proba(
        df
    )[0][1]

    prediction = model.predict(
        df
    )[0]

    risk = get_risk_level(
        probability
    )

    return {

        "prediction": int(prediction),

        "probability":
            round(
                probability * 100,
                2
            ),

        "risk_level": risk
    }

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(model):

    rf_model = model.named_steps[
        "classifier"
    ]

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            rf_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return importance

# ==========================================================
# SHAP SUMMARY
# ==========================================================

def generate_shap_summary(
        model,
        sample_df
):

    transformed = model.named_steps[
        "preprocessor"
    ].transform(sample_df)

    classifier = model.named_steps[
        "classifier"
    ]

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        transformed
    )

    plt.figure(
        figsize=(10, 6)
    )

    shap.summary_plot(
        shap_values,
        transformed,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "shap_summary.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# SHAP WATERFALL
# ==========================================================

def generate_shap_waterfall(
        model,
        sample_df
):

    transformed = model.named_steps[
        "preprocessor"
    ].transform(sample_df)

    classifier = model.named_steps[
        "classifier"
    ]

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer(
        transformed
    )

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    plt.savefig(
        "shap_waterfall.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# CROSS VALIDATION
# ==========================================================

def cross_validation(
        model,
        X,
        y
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    print(
        "\nAverage CV ROC-AUC:",
        scores.mean()
    )

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train_model(
        DATA_PATH
    )