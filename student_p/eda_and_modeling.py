"""
eda_and_modeling.py
EDA + Model Training for Student Placement Prediction
Author: Project by Smrity Shreya
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  STUDENT PLACEMENT PREDICTION — EDA & MODELING")
print("=" * 60)

df = pd.read_csv("placement_data.csv")
print(f"\n[INFO] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── 2. Basic Checks ───────────────────────────────────────────────────────────
print("── Missing Values ─────────────────────────────")
print(df.isnull().sum())

print("\n── Statistical Description ───────────────────")
print(df.describe().round(3).to_string())

print("\n── Target Distribution ───────────────────────")
vc = df["Placed"].value_counts()
print(f"  Placed (1): {vc.get(1, 0)}   Not Placed (0): {vc.get(0, 0)}")

# ── 3. EDA Plots ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")

# Plot 1 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax,
    vmin=-1,
    vmax=1,
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", dpi=150)
plt.show()
print("\n[SAVED] plot_correlation_heatmap.png")

# Plot 2 — CGPA Distribution by Placement Status
fig, ax = plt.subplots(figsize=(8, 4))
colors = {0: "#e74c3c", 1: "#2ecc71"}
labels = {0: "Not Placed", 1: "Placed"}
for label, grp in df.groupby("Placed"):
    sns.kdeplot(
        grp["CGPA"],
        ax=ax,
        fill=True,
        alpha=0.45,
        color=colors[label],
        label=labels[label],
        linewidth=2,
    )
ax.set_title("CGPA Distribution by Placement Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("CGPA", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plot_cgpa_distribution.png", dpi=150)
plt.show()
print("[SAVED] plot_cgpa_distribution.png")

# ── 4. Feature / Target Split & Train-Test Split ──────────────────────────────
FEATURES = ["CGPA", "Internships", "Projects", "Technical_Skills_Score", "Core_Backlogs"]
TARGET = "Placed"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[INFO] Train size: {len(X_train)}   Test size: {len(X_test)}")

# ── 5. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(
                               n_estimators=200,
                               max_depth=8,
                               random_state=42,
                               class_weight="balanced",
                           ),
}

results = {}
print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])

    results[name] = {"model": model, "accuracy": acc, "cm": cm, "report": report}

    print(f"\n── {name} ────────────────────────")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"plot_cm_{safe_name}.png", dpi=150)
    plt.show()
    print(f"[SAVED] plot_cm_{safe_name}.png")

# ── 7. Compare & Pick Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["accuracy"])
print(f"\n[WINNER] Best model → {best_name}  "
      f"(Accuracy: {results[best_name]['accuracy']*100:.2f}%)")

best_model = results["Random Forest"]["model"]   # per requirement: save RF

# ── 8. Random Forest — Feature Importances ───────────────────────────────────
importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.show()
print("[SAVED] plot_feature_importance.png")

# ── 9. Persist Artefacts ──────────────────────────────────────────────────────
with open("placement_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\n[SAVED] placement_model.pkl")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("[SAVED] scaler.pkl")

print("\n✅  Pipeline complete. All artefacts ready for Streamlit app.\n")
