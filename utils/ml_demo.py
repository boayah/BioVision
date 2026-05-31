import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS = [
    "Brightness",
    "Contrast",
    "Blur Score",
    "Overexposure %",
    "Underexposure %",
    "Edge Density %",
    "Coverage %",
]


def train_and_evaluate(labeled_df, model_type="Random Forest"):
    X = labeled_df[FEATURE_COLS].values
    y = labeled_df["Label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n = len(X)
    test_size = 2 if n < 8 else 0.25

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importances = model.feature_importances_
    else:
        model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if len(le.classes_) == 2:
            importances = np.abs(model.coef_[0])
        else:
            importances = np.mean(np.abs(model.coef_), axis=0)

    acc = accuracy_score(y_test, y_pred)
    present_labels = np.unique(np.concatenate([y_train, y_test]))
    present_class_names = le.inverse_transform(present_labels)

    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "accuracy": acc,
        "classification_report": report,
        "importances": importances,
        "feature_names": FEATURE_COLS,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def make_feature_importance_chart(importances, feature_names):
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0"] * len(feature_names)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importances")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
