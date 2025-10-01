# -----------------------------
# Task 2: Customer Churn Prediction with 4 Visualizations
# -----------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# -----------------------------
# 1. Setup
# -----------------------------
OUTPUT_DIR = "outputs_task2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("customer_churn.csv")
target = "Churn"

X = df.drop(columns=[target, "customerID"])
y = df[target].map({"Yes": 1, "No": 0})

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Train Models & Compare ROC AUC
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

roc_scores = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    roc_scores[name] = roc_auc_score(y_test, y_prob)

# -----------------------------
# 4. Select Best Model
# -----------------------------
best_model_name = max(roc_scores, key=roc_scores.get)
print(f"Best model based on ROC AUC: {best_model_name} (ROC AUC = {roc_scores[best_model_name]:.4f})")

best_model = models[best_model_name]
best_pipe = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
best_pipe.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

# -----------------------------
# 6. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_model_name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_confusion_matrix.png"))
plt.show()

# -----------------------------
# 7. ROC Curve
# -----------------------------
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title(f"{best_model_name} - ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_roc_curve.png"))
plt.show()

# -----------------------------
# 8. Churn Distribution
# -----------------------------
plt.figure(figsize=(5,4))
sns.countplot(x=target, data=df, palette="pastel")
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "churn_distribution.png"))
plt.show()

# -----------------------------
# 9. Feature Importance / Coefficients
# -----------------------------
if hasattr(best_model, "feature_importances_"):  # Tree-based
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    feature_names = numeric_cols + ohe.get_feature_names_out(categorical_cols).tolist()
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
elif hasattr(best_model, "coef_"):  # Logistic Regression
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    feature_names = numeric_cols + ohe.get_feature_names_out(categorical_cols).tolist()
    importances = pd.Series(abs(best_model.coef_[0]), index=feature_names)  # absolute value

top_features = importances.sort_values(ascending=False).head(15)
plt.figure(figsize=(8,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title(f"{best_model_name} - Top 15 Features Affecting Churn")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_feature_importance.png"))
plt.show()

# -----------------------------
# 10. Churn Probabilities & High-Risk Customers
# -----------------------------
df["Churn_Probability"] = best_pipe.predict_proba(X)[:, 1]
df.to_csv(os.path.join(OUTPUT_DIR, "customer_churn_probabilities.csv"), index=False)

high_risk_customers = df[df["Churn_Probability"] > 0.7]
high_risk_customers.to_csv(os.path.join(OUTPUT_DIR, "high_risk_customers.csv"), index=False)
print(f"Number of high-risk customers (>0.7 probability): {high_risk_customers.shape[0]}")

# -----------------------------
# 11. Model Summary
# -----------------------------
summary = pd.DataFrame({"ROC AUC": roc_scores}).T
summary.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"))
print("\nModel comparison summary:\n", summary)

# -----------------------------
# 12. Business Recommendations
# -----------------------------
print("\nBusiness Recommendations:")
print("- Target high-risk customers with retention offers (discounts, loyalty benefits).")
print("- Review plans with high churn probability and improve customer experience.")
print("- Use churn probability to prioritize customer support for retention campaigns.")
