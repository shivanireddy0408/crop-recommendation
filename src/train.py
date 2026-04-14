"""
=============================================================
  Smart Crop Recommendation System — Training Pipeline
=============================================================
Author  : Senior Data Scientist
Purpose : Full ML pipeline — EDA, preprocessing, model
          training, comparison, and best-model export.
=============================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/Crop_recommendation.csv"
MODEL_DIR    = "model"
NOTEBOOK_DIR = "notebook"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SMART CROP RECOMMENDATION SYSTEM — TRAINING PIPELINE")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA]  Shape : {df.shape}")
print(f"[DATA]  Crops : {sorted(df['label'].unique())}\n")
print(df.head())

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA & DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Descriptive Statistics ──────────────────────────────")
print(df.describe().round(2))

print("\n── Missing Values ──────────────────────────────────────")
print(df.isnull().sum())

print("\n── Class Distribution ──────────────────────────────────")
print(df["label"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# 3a. Correlation heatmap
print("\n[VIZ] Generating correlation heatmap…")
plt.figure(figsize=(10, 7))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# 3b. Feature distributions
print("[VIZ] Generating feature distributions…")
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
palette = sns.color_palette("husl", len(FEATURES))
for i, feat in enumerate(FEATURES):
    axes[i].hist(df[feat], bins=35, color=palette[i], edgecolor="white", alpha=0.85)
    axes[i].set_title(feat, fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=10)
    axes[i].set_ylabel("Frequency", fontsize=10)
    axes[i].grid(axis="y", linestyle="--", alpha=0.4)
axes[-1].set_visible(False)
fig.suptitle("Feature Distributions", fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 3c. Boxplots by crop (temperature & rainfall)
print("[VIZ] Generating boxplots…")
fig, axes = plt.subplots(1, 2, figsize=(22, 7))
crop_order = sorted(df["label"].unique())
sns.boxplot(data=df, x="label", y="temperature", order=crop_order,
            palette="Set2", ax=axes[0])
axes[0].set_title("Temperature Distribution per Crop", fontsize=14, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_xlabel("")

sns.boxplot(data=df, x="label", y="rainfall", order=crop_order,
            palette="Set3", ax=axes[1])
axes[1].set_title("Rainfall Distribution per Crop", fontsize=14, fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/boxplots_by_crop.png", dpi=150, bbox_inches="tight")
plt.close()

print("[VIZ] All visualisations saved to notebook/")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Preprocessing ───────────────────────────────────────")

X = df[FEATURES]
y = df["label"]

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"[PRE]  Label classes ({len(le.classes_)}): {list(le.classes_)}")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"[PRE]  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# Feature scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model Training ──────────────────────────────────────")

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                             X_train_sc, X_test_sc),
    "Decision Tree":       (DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
                             X_train,    X_test),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, max_depth=15,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
                             X_train,    X_test),
}

results = {}
trained_models = {}

for name, (model, Xtr, Xte) in models.items():
    print(f"\n  ▸ Training {name}…")
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)

    # 5-fold cross-validation on train set
    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")

    results[name] = {
        "accuracy":  acc,
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "y_pred":    y_pred,
        "model":     model,
        "Xte":       Xte,
    }
    trained_models[name] = model
    print(f"    Test Accuracy : {acc:.4f}")
    print(f"    CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ] Generating model comparison chart…")
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] for m in model_names]
cv_means    = [results[m]["cv_mean"]  for m in model_names]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Test Accuracy",  color="#2ecc71", edgecolor="white")
bars2 = ax.bar(x + width/2, cv_means,   width, label="CV Mean (5-fold)", color="#3498db", edgecolor="white")

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/model_comparison.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. BEST MODEL — DETAILED EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda m: results[m]["accuracy"])
best_info = results[best_name]
best_model = best_info["model"]

print("\n" + "="*60)
print(f"  BEST MODEL: {best_name}  (Accuracy: {best_info['accuracy']:.4f})")
print("="*60)

print("\n── Classification Report ──────────────────────────────")
print(classification_report(
    y_test, best_info["y_pred"],
    target_names=le.classes_
))

# Confusion matrix
print("[VIZ] Generating confusion matrix…")
cm = confusion_matrix(y_test, best_info["y_pred"])
fig, ax = plt.subplots(figsize=(16, 13))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

print("\n[VIZ] Generating feature importance chart…")
fig, ax = plt.subplots(figsize=(9, 5))
colors = sns.color_palette("viridis", len(FEATURES))
bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="white")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)
for bar in bars:
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{NOTEBOOK_DIR}/feature_importance.png", dpi=150)
plt.close()

print("[VIZ] All charts saved to notebook/")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Saving Artifacts ────────────────────────────────────")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")
joblib.dump(le,         f"{MODEL_DIR}/label_encoder.pkl")

print(f"  ✔  {MODEL_DIR}/best_model.pkl     ({best_name})")
print(f"  ✔  {MODEL_DIR}/scaler.pkl")
print(f"  ✔  {MODEL_DIR}/label_encoder.pkl")

# Save model summary
summary_lines = [
    "CROP RECOMMENDATION SYSTEM — MODEL SUMMARY",
    "=" * 45,
    f"Best Model   : {best_name}",
    f"Test Accuracy: {best_info['accuracy']:.4f}",
    f"CV Mean      : {best_info['cv_mean']:.4f} ± {best_info['cv_std']:.4f}",
    "",
    "All Models:",
]
for m in model_names:
    summary_lines.append(
        f"  {m:25s}  acc={results[m]['accuracy']:.4f}  cv={results[m]['cv_mean']:.4f}"
    )

with open(f"{MODEL_DIR}/model_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
print(f"  ✔  {MODEL_DIR}/model_summary.txt")

print("\n✅  Training pipeline complete!\n")
