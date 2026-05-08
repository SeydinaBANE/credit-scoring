"""
Entraînement XGBoost — Scoring de crédit alternatif
Inclut : validation croisée, SHAP, calibration, sauvegarde modèle
"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, brier_score_loss,
)
import xgboost as xgb

DATA_DIR  = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_data():
    X = pd.read_csv(DATA_DIR / "features.csv", index_col="client_id")
    y = pd.read_csv(DATA_DIR / "labels.csv",   index_col="client_id").squeeze()
    print(f"Données chargées : {X.shape[0]} clients, {X.shape[1]} features")
    return X, y


# ══════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════

def train_model(X: pd.DataFrame, y: pd.Series):

    # ── Split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nSplit : {len(X_train)} train / {len(X_test)} test")

    # ── Modèle XGBoost ─────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    # ── Entraînement avec early stopping ──────────────────────────
    print("\nEntraînement XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Validation croisée ─────────────────────────────────────────
    print("\nValidation croisée (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        X, y, cv=cv, scoring="roc_auc"
    )
    print(f"AUC CV : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Calibration des probabilités ──────────────────────────────
    print("\nCalibration des probabilités (isotonic)...")
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    return calibrated, model, X_train, X_test, y_train, y_test, cv_scores


# ══════════════════════════════════════════════════════════════════════
# 3. ÉVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(calibrated, X_test, y_test):
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc    = roc_auc_score(y_test, y_proba)
    brier  = brier_score_loss(y_test, y_proba)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Défaut", "Bon payeur"])

    print("\n" + "="*50)
    print("RÉSULTATS D'ÉVALUATION")
    print("="*50)
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Brier    : {brier:.4f}  (0 = parfait)")
    print(f"\nMatrice de confusion :")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{report}")

    metrics = {
        "auc_roc": round(auc, 4),
        "brier_score": round(brier, 4),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "trained_at": datetime.now().isoformat(),
    }
    return metrics, y_proba


# ══════════════════════════════════════════════════════════════════════
# 4. SHAP — IMPORTANCE & EXPLICATION
# ══════════════════════════════════════════════════════════════════════

def compute_shap(model, X_train, X_test):
    print("\nCalcul des valeurs SHAP...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Importance globale
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)

    print("\nTop 15 features les plus importantes (SHAP) :")
    print(mean_abs_shap.head(15).round(4).to_string())

    return explainer, shap_values, mean_abs_shap


def explain_client(explainer, X_client: pd.DataFrame, top_n=5) -> dict:
    """
    Retourne les top_n facteurs SHAP pour un client donné.
    Utilisé par le nœud LangGraph pour générer l'explication LLM.
    """
    sv = explainer.shap_values(X_client)[0]
    factors = pd.Series(sv, index=X_client.columns).sort_values(key=abs, ascending=False)
    top = factors.head(top_n)
    return {
        feat: {
            "shap_value": round(float(val), 4),
            "feature_value": round(float(X_client[feat].iloc[0]), 4),
            "impact": "positif" if val > 0 else "négatif",
        }
        for feat, val in top.items()
    }


# ══════════════════════════════════════════════════════════════════════
# 5. CONVERSION EN SCORE 0–1000
# ══════════════════════════════════════════════════════════════════════

def proba_to_score(proba: float) -> int:
    """Convertit une probabilité [0,1] en score crédit [300, 900]."""
    return int(300 + proba * 600)


# ══════════════════════════════════════════════════════════════════════
# 6. PRÉDICTION (utilisé par l'agent LangGraph)
# ══════════════════════════════════════════════════════════════════════

def predict_score(features: dict) -> tuple[int, dict]:
    """
    Prédit le score crédit d'un client à partir de ses features.
    Retourne (score_0_1000, shap_factors).
    """
    model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    X_client = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    proba    = model.predict_proba(X_client)[:, 1][0]
    score    = proba_to_score(proba)
    factors  = explain_client(explainer, X_client)

    return score, factors


# ══════════════════════════════════════════════════════════════════════
# 7. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════

def save_artifacts(calibrated, explainer, feature_names, metrics, mean_abs_shap):
    joblib.dump(calibrated,    MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(explainer,     MODEL_DIR / "shap_explainer.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mean_abs_shap.to_csv(MODEL_DIR / "shap_importance.csv", header=["importance"])

    print(f"\nArtifacts sauvegardés dans {MODEL_DIR}/")
    print("  xgboost_model.pkl")
    print("  shap_explainer.pkl")
    print("  feature_names.pkl")
    print("  metrics.json")
    print("  shap_importance.csv")


# ══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    X, y = load_data()

    calibrated, model, X_train, X_test, y_train, y_test, cv_scores = train_model(X, y)

    metrics, y_proba = evaluate_model(calibrated, X_test, y_test)
    metrics["cv_auc_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_auc_std"]  = round(cv_scores.std(), 4)

    explainer, shap_values, mean_abs_shap = compute_shap(model, X_train, X_test)

    save_artifacts(calibrated, explainer, list(X.columns), metrics, mean_abs_shap)

    # ── Test predict_score sur un client exemple ───────────────────
    print("\nTest predict_score sur un client exemple...")
    sample_features = X_test.iloc[0].to_dict()
    score, factors  = predict_score(sample_features)
    print(f"Score     : {score}/900")
    print(f"Facteurs  : {json.dumps(factors, indent=2, ensure_ascii=False)}")