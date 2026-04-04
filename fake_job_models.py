
# FAKE JOB DETECTION - 3 MODEL TRAINING & EVALUATION
# Dataset: Kaggle Fake Job Postings
# Models: Logistic Regression, Random Forest, Naive Bayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# STEP 1: LOAD DATASET

df = pd.read_csv("merged_job_postings.csv")
print("Dataset shape:", df.shape)
print("Fraud distribution:\n", df["fraudulent"].value_counts())


# STEP 2: TEXT PREPROCESSING
# Fill missing values and combine key text columns into one
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

X = df["combined_text"]
y = df["fraudulent"]


# STEP 3: TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# STEP 4: DEFINE MODELS
# Each model is wrapped in a Pipeline with TF-IDF vectorizer
# class_weight='balanced' handles the imbalanced dataset

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
    ]),

    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("clf", MultinomialNB(alpha=0.1))
        # Note: MultinomialNB does not support class_weight
        # Use alpha tuning to manage imbalance
    ])
}

# STEP 5: TRAIN, PREDICT & EVALUATE ALL 3 MODELS

results = {}

for name, pipeline in models.items():
    print(f"\n{'='*50}")
    print(f"  {name}")
    print("="*50)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec  = recall_score(y_test, y_pred, pos_label=1)
    f1   = f1_score(y_test, y_pred, pos_label=1)

    results[name] = {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4)
    }

    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()


# STEP 6: COMPARATIVE SUMMARY TABLE

print("\n\n" + "="*60)
print("  COMPARATIVE MODEL ANALYSIS")
print("="*60)
summary_df = pd.DataFrame(results).T
print(summary_df.to_string())

# Plot comparison bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
summary_df[metrics].plot(kind="bar", figsize=(10, 6), colormap="Set2", edgecolor="black")
plt.title("Model Comparison — Fake Job Detection")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# STEP 7: FINAL MODEL SELECTION
# Select model with best F1 score on the fake class

best_model_name = summary_df["F1 Score"].idxmax()
print(f"\n>>> Best Model: {best_model_name}")
print(f">>> F1 Score (Fake class): {summary_df.loc[best_model_name, 'F1 Score']}")
print("\nRationale: F1 score is prioritized over accuracy because the")
print("dataset is imbalanced — we care more about catching fake jobs (recall)")
print("while not flagging too many real ones (precision).")