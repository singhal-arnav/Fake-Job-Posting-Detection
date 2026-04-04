# FAKE JOB DETECTION — MODEL TRAINING & EVALUATION
# Dataset: Kaggle Fake Job Postings (merged_job_postings.csv)
# Models: Logistic Regression, Random Forest, Naive Bayes
# Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix,
#             Precision-Recall Curve, 5-Fold CV, Feature Importance

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    PrecisionRecallDisplay,
)

SEED = 42
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110


# ── STEP 1: LOAD DATASET ──────────────────────────────────────────────────────

df = pd.read_csv("merged_job_postings.csv")
print("Dataset shape:", df.shape)
print("Fraud distribution:\n", df["fraudulent"].value_counts())


# ── STEP 2: TEXT PRE-PROCESSING ───────────────────────────────────────────────
# Combine five text columns; apply cleaning (lowercase, strip HTML & punctuation)

def clean_text(text):
    """Lowercase, strip HTML tags, non-alpha characters, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text

text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["text"] = df[text_cols].apply(lambda row: " ".join(row.values), axis=1)
df["text"] = df["text"].apply(clean_text)

print("\nSample cleaned text (first 200 chars):")
print(df["text"].iloc[0][:200])


# ── STEP 3: TRAIN / TEST SPLIT ────────────────────────────────────────────────
# 80/20 stratified split to preserve the ~4.8 % minority (fraud) class ratio

X = df["text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
print("Class distribution in test set:")
print(y_test.value_counts(normalize=True).rename({0: "real", 1: "fake"}).to_string())


# ── STEP 4: TF-IDF VECTORISATION ──────────────────────────────────────────────
# Fit on train only to prevent data leakage.
# 20 k features, unigrams + bigrams, log-scaled TF, English stop words removed.

tfidf = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\nTF-IDF matrix — train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")


# ── STEP 5: DEFINE & TRAIN MODELS ─────────────────────────────────────────────
# class_weight='balanced' compensates for the imbalanced target distribution.
# MultinomialNB does not support class_weight; alpha is tuned instead.

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED
    ),
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    print("done.")


# ── STEP 6: EVALUATE ALL MODELS ───────────────────────────────────────────────

def evaluate_model(name, model, X_te, y_te):
    """Return a dict of scalar metrics for a trained classifier."""
    y_pred = model.predict(X_te)
    return {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, pos_label=1, zero_division=0),
        "Recall"   : recall_score(y_te, y_pred, pos_label=1, zero_division=0),
        "F1 Score" : f1_score(y_te, y_pred, pos_label=1, zero_division=0),
    }

results = [
    evaluate_model(name, model, X_test_tfidf, y_test)
    for name, model in trained_models.items()
]

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n" + "="*50)
print("  METRICS SUMMARY")
print("="*50)
print(results_df.to_string())


# ── STEP 7: DETAILED CLASSIFICATION REPORTS ───────────────────────────────────

for name, model in trained_models.items():
    y_pred = model.predict(X_test_tfidf)
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# ── STEP 8: COMPARISON BAR CHART ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(results_df.columns))
width  = 0.25
colors = sns.color_palette("Set2", 3)

for i, (model_name, row) in enumerate(results_df.iterrows()):
    bars = ax.bar(x + i * width, row.values, width, label=model_name, color=colors[i])
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

ax.set_xticks(x + width)
ax.set_xticklabels(results_df.columns, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics (Test Set)")
ax.legend(loc="upper right")
sns.despine()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()


# ── STEP 9: CONFUSION MATRICES ────────────────────────────────────────────────
# False negatives (missed fraud) are more costly than false positives here.

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices (Test Set)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()


# ── STEP 10: PRECISION–RECALL CURVES ─────────────────────────────────────────
# More informative than ROC for imbalanced datasets — focuses on the minority class.

fig, ax = plt.subplots(figsize=(7, 5))
colors = sns.color_palette("Set2", 3)
for (name, model), color in zip(trained_models.items(), colors):
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, name=name, ax=ax, color=color
    )

ax.set_title("Precision–Recall Curves — All Models (Fraud Class)")
ax.legend(loc="upper right")
sns.despine()
plt.tight_layout()
plt.savefig("pr_curves.png", dpi=150)
plt.show()


# ── STEP 11: 5-FOLD STRATIFIED CROSS-VALIDATION ───────────────────────────────
# Scored by F1 (fraud class) for a robust generalisation estimate.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", clf),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:<25}  CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Box-plot of CV F1 scores
fig, ax = plt.subplots(figsize=(7, 4))
pd.DataFrame(cv_results).boxplot(ax=ax)
ax.set_ylabel("F1 Score (fraud class)")
ax.set_title("5-Fold CV F1 Scores by Model")
sns.despine()
plt.tight_layout()
plt.savefig("cv_f1_boxplot.png", dpi=150)
plt.show()


# ── STEP 12: COMPARATIVE SUMMARY TABLE ────────────────────────────────────────
# Ranked by F1 score on the fraud class.

cv_f1_means = {name: scores.mean() for name, scores in cv_results.items()}
results_df["CV F1 (mean)"] = results_df.index.map(cv_f1_means)
summary = results_df.sort_values("F1 Score", ascending=False).round(4)

print("\n" + "="*60)
print("  COMPARATIVE MODEL ANALYSIS — RANKED BY F1 SCORE")
print("="*60)
print(summary.to_string())


# ── STEP 13: FINAL MODEL SELECTION ────────────────────────────────────────────
# Logistic Regression selected: best F1, strong PR curve, interpretable, fast.

best_name  = summary.index[0]          # top-ranked by F1
best_model = trained_models[best_name]

y_pred = best_model.predict(X_test_tfidf)

print(f"\n{'='*50}")
print(f"  Final Model: {best_name}")
print(f"{'='*50}")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, pos_label=1):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, pos_label=1):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"{'='*50}\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# ── STEP 14: FEATURE IMPORTANCE (Logistic Regression) ─────────────────────────
# Top positive coefficients → fraud signal; top negative → real signal.

if hasattr(best_model, "coef_"):
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = best_model.coef_[0]
    top_n = 20

    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(feature_names[top_pos_idx][::-1], coef[top_pos_idx][::-1], color="#e74c3c")
    axes[0].set_title("Top words predicting FRAUD\n(positive coefficients)", fontsize=11)
    axes[0].set_xlabel("Coefficient value")

    axes[1].barh(feature_names[top_neg_idx], coef[top_neg_idx], color="#2ecc71")
    axes[1].set_title("Top words predicting REAL\n(negative coefficients)", fontsize=11)
    axes[1].set_xlabel("Coefficient value")

    plt.suptitle(f"{best_name} — Most Informative Features", fontsize=13)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
