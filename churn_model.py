import os
import warnings
import json
import joblib
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATASET = "European_Bank.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


class BankChurnTrainer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.pipeline = None
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.feature_names = None
        self.preprocessor = None
        self.X_test = None
        self.y_test = None

    # ============================================================
    # Load Dataset
    # ============================================================

    def load_dataset(self):
        logging.info("Loading Dataset...")
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Dataset Shape : {self.df.shape}")
        return self.df

    # ============================================================
    # Validate Dataset
    # ============================================================

    def validate_dataset(self):
        required = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing Columns : {missing}")
        logging.info("Dataset Validation Passed")

    # ============================================================
    # Data Cleaning
    # ============================================================

    def clean_dataset(self):
        logging.info("Cleaning Dataset...")
        remove = ["RowNumber", "CustomerId", "Surname"]
        self.df.drop(
            columns=[c for c in remove if c in self.df.columns],
            inplace=True,
            errors="ignore",
        )
        self.df.drop_duplicates(inplace=True)
        for column in self.df.columns:
            if self.df[column].dtype == object or pd.api.types.is_string_dtype(self.df[column]):
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            else:
                self.df[column].fillna(self.df[column].median(), inplace=True)
        logging.info("Cleaning Complete")

    # ============================================================
    # Feature Engineering
    # ============================================================

    def feature_engineering(self):
        logging.info("Creating Features...")
        self.df["Balance_to_Salary_Ratio"] = self.df["Balance"] / (self.df["EstimatedSalary"] + 1)
        self.df["Products_Per_Tenure"] = self.df["NumOfProducts"] / (self.df["Tenure"] + 1)
        self.df["Age_Tenure_Interaction"] = self.df["Age"] * self.df["Tenure"]
        self.df["Customer_Engagement_Score"] = (
            self.df["IsActiveMember"] * 2 + self.df["HasCrCard"] + self.df["NumOfProducts"]
        )
        logging.info("Feature Engineering Complete")

    # ============================================================
    # Dataset Summary
    # ============================================================

    def summary(self):
        logging.info("=" * 60)
        logging.info("Dataset Summary")
        logging.info("=" * 60)
        print(self.df.info())
        print(self.df.describe())
        print(self.df.head())

    # ============================================================
    # Missing Values
    # ============================================================

    def missing_values(self):
        print(self.df.isna().sum().sort_values(ascending=False))

    # ============================================================
    # Exploratory Data Analysis
    # ============================================================

    def perform_eda(self):
        logging.info("Generating EDA...")
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x="Exited")
        plt.title("Churn Distribution")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="Geography", hue="Exited")
        plt.title("Geography-wise Churn")
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x="Gender", hue="Exited")
        plt.title("Gender-wise Churn")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.histplot(self.df["Age"], bins=30, kde=True)
        plt.title("Age Distribution")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.histplot(self.df["Balance"], bins=30, kde=True)
        plt.title("Balance Distribution")
        plt.show()

        plt.figure(figsize=(14, 8))
        sns.heatmap(self.df.corr(numeric_only=True), cmap="coolwarm", annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

    # ============================================================
    # Data Preprocessing
    # ============================================================

    def preprocess_data(self):
        logging.info("Preparing data for training...")
        X = self.df.drop(columns=["Exited"])
        y = self.df["Exited"]
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
        self.feature_names = X.columns.tolist()
        return X_train, X_test, y_train, y_test

    # ============================================================
    # Build Models
    # ============================================================

    def build_models(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss")
        return models

    # ============================================================
    # Train Models
    # ============================================================

    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        models = self.build_models()
        results = []
        best_auc = 0
        for name, classifier in models.items():
            logging.info(f"Training {name}")
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", classifier),
                ]
            )
            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_test)
            probability = pipeline.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(y_test, prediction)
            recall = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)
            auc_value = roc_auc_score(y_test, probability)
            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": auc_value,
            })
            if auc_value > best_auc:
                best_auc = auc_value
                self.best_model = pipeline
                self.best_model_name = name
                self.X_test = X_test
                self.y_test = y_test
        results = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
        logging.info("\nModel Comparison")
        print(results)
        return results

    # ============================================================
    # Cross Validation
    # ============================================================

    def cross_validation(self):
        logging.info("Running Cross Validation")
        X = self.df.drop(columns=["Exited"])
        y = self.df["Exited"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.best_model, X, y, cv=cv, scoring="roc_auc")
        logging.info(f"Average ROC-AUC : {scores.mean():.4f}")
        print(scores)

    # ============================================================
    # Hyperparameter Tuning
    # ============================================================

    def hyperparameter_tuning(self):
        if self.best_model_name != "Random Forest":
            logging.info("Skipping tuning (only Random Forest).")
            return
        logging.info("Running GridSearchCV...")
        parameters = {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
        }
        X_train, _, y_train, _ = self.preprocess_data()
        grid = GridSearchCV(self.best_model, parameters, cv=5, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        self.best_model = grid.best_estimator_
        logging.info(f"Best Parameters : {grid.best_params_}")

    # ============================================================
    # Model Evaluation
    # ============================================================

    def evaluate_model(self):
        logging.info("Evaluating Best Model...")
        prediction = self.best_model.predict(self.X_test)
        probability = self.best_model.predict_proba(self.X_test)[:, 1]
        self.metrics = {
            "Accuracy": accuracy_score(self.y_test, prediction),
            "Precision": precision_score(self.y_test, prediction),
            "Recall": recall_score(self.y_test, prediction),
            "F1 Score": f1_score(self.y_test, prediction),
            "ROC AUC": roc_auc_score(self.y_test, probability),
        }
        print("\nClassification Report\n")
        print(classification_report(self.y_test, prediction))
        return prediction, probability

    # ============================================================
    # Confusion Matrix
    # ============================================================

    def plot_confusion_matrix(self):
        prediction = self.best_model.predict(self.X_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(self.y_test, prediction, cmap="Blues", ax=ax)
        plt.title("Confusion Matrix")
        plt.show()

    # ============================================================
    # ROC Curve
    # ============================================================

    def plot_roc_curve(self):
        probability = self.best_model.predict_proba(self.X_test)[:, 1]
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(self.y_test, probability, ax=ax)
        plt.title("ROC Curve")
        plt.show()

    # ============================================================
    # Feature Importance
    # ============================================================

    def feature_importance(self):
        classifier = self.best_model.named_steps["classifier"]
        preprocessor = self.best_model.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            importance = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importance = np.abs(classifier.coef_[0])
        else:
            return
        importance_df = (
            pd.DataFrame({"Feature": names, "Importance": importance})
            .sort_values("Importance", ascending=False)
        )
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
        plt.title("Top 20 Feature Importance")
        plt.show()
        return importance_df

    # ============================================================
    # SHAP
    # ============================================================

    def shap_analysis(self):
        try:
            classifier = self.best_model.named_steps["classifier"]
            preprocessor = self.best_model.named_steps["preprocessor"]
            transformed = preprocessor.transform(self.X_test)
            feature_names = preprocessor.get_feature_names_out()
            if hasattr(classifier, "feature_importances_"):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(transformed)
                shap.summary_plot(shap_values, transformed, feature_names=feature_names)
            else:
                logging.info("SHAP skipped for this model.")
        except Exception as e:
            logging.warning(e)

    # ============================================================
    # Save Artifacts
    # ============================================================

    def save_artifacts(self):
        logging.info("Saving model artifacts...")
        joblib.dump(self.best_model, os.path.join(MODEL_DIR, "churn_model.pkl"))
        joblib.dump(self.X_test, os.path.join(MODEL_DIR, "X_test.pkl"))
        joblib.dump(self.y_test, os.path.join(MODEL_DIR, "y_test.pkl"))
        joblib.dump(self.feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
        joblib.dump(self.best_model_name, os.path.join(MODEL_DIR, "model_name.pkl"))
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as file:
            json.dump(self.metrics, file, indent=4)
        logging.info("Artifacts Saved Successfully.")

    # ============================================================
    # Complete Pipeline
    # ============================================================

    def run_pipeline(self):
        self.load_dataset()
        self.validate_dataset()
        self.clean_dataset()
        self.feature_engineering()
        self.summary()
        self.missing_values()
        self.perform_eda()
        self.train_models()
        self.cross_validation()
        self.hyperparameter_tuning()
        self.evaluate_model()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.feature_importance()
        self.shap_analysis()
        self.save_artifacts()


if __name__ == "__main__":
    trainer = BankChurnTrainer(DATASET)
    trainer.run_pipeline()
