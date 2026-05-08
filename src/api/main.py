"""
API FastAPI — CréditScore AI SaaS
Endpoints : scoring, batch, health, docs
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import time
import uuid
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from src.config import TIER_URBAIN, SCORE_OPERATEUR, assess_risk as _assess_risk

load_dotenv()

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# ══════════════════════════════════════════════════════════════════════
# 1. APP & MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CréditScore AI",
    description="""
## API de scoring de crédit alternatif pour les non-bankés en Afrique de l'Ouest

Analyse les données **Mobile Money** (Wave, Orange Money) et **Telco** pour produire :
- Un **score crédit 300–900**
- Une **explication en français** générée par Claude AI
- Une **décision** (approuvé / révision / refusé) avec montant et durée suggérés

### Authentification
Toutes les routes nécessitent un header `X-API-Key`.
    """,
    version="1.0.0",
    contact={"name": "CréditScore AI", "email": "api@creditscore.ai"},
    license_info={"name": "Propriétaire"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
# 2. AUTHENTIFICATION PAR API KEY
# ══════════════════════════════════════════════════════════════════════

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# En prod : stocker dans DB avec tiers, quotas, etc.
VALID_API_KEYS = {
    os.getenv("API_KEY_DEV",  "cs-dev-key-123"):  {"tier": "dev",  "limit": 100},
    os.getenv("API_KEY_PROD", "cs-prod-key-456"): {"tier": "prod", "limit": 10000},
}

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide. Passez X-API-Key dans le header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return VALID_API_KEYS[api_key]


# ══════════════════════════════════════════════════════════════════════
# 3. SCHEMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════════

class MobileMoney(BaseModel):
    nb_transactions_total:     int   = Field(..., ge=0, le=10000,   description="Nombre total de transactions")
    montant_moyen_xof:         float = Field(..., ge=0, le=5000000, description="Montant moyen par transaction (XOF)")
    regularite_mensuelle:      float = Field(..., ge=0, le=1,       description="Régularité mensuelle [0–1]")
    ratio_epargne:             float = Field(0.0, ge=0, le=1,       description="Part des transactions d'épargne")
    tendance_activite:         float = Field(0.0, ge=-1, le=2,      description="Tendance récente vs passée")
    nb_tx_30j:                 int   = Field(0,   ge=0,             description="Transactions sur 30 jours")
    nb_tx_90j:                 int   = Field(0,   ge=0,             description="Transactions sur 90 jours")
    recence_jours:             int   = Field(30,  ge=0, le=365,     description="Jours depuis dernière transaction")

class Telco(BaseModel):
    nb_recharges_6m:           int   = Field(..., ge=0, le=365,   description="Nombre de recharges sur 6 mois")
    regularite_recharge:       float = Field(..., ge=0, le=1,     description="Régularité des recharges [0–1]")
    jours_actif_30j:           int   = Field(..., ge=0, le=30,    description="Jours d'activité sur 30 jours")
    nb_contacts_uniques_30j:   int   = Field(10,  ge=0, le=500,   description="Contacts uniques sur 30 jours")
    duree_moy_appel_min:       float = Field(2.0, ge=0, le=60,    description="Durée moyenne des appels (min)")
    a_internet_mobile:         int   = Field(0,   ge=0, le=1,     description="Accès internet mobile [0/1]")
    montant_moy_recharge:      float = Field(1000, ge=0,          description="Montant moyen de recharge (XOF)")

class ProfilClient(BaseModel):
    age:              int    = Field(..., ge=18, le=100, description="Âge du client")
    anciennete_mois:  int    = Field(..., ge=1,  le=240, description="Ancienneté avec l'opérateur (mois)")
    a_emprunte_avant: int    = Field(0,   ge=0,  le=1,   description="A déjà emprunté [0/1]")
    region:           Literal["Dakar","Thiès","Saint-Louis","Kaolack","Ziguinchor","Diourbel","Tambacounda","Kolda"] = Field("Dakar", description="Région")
    operateur:        Literal["Wave","Orange","Free","Expresso"] = Field("Orange", description="Opérateur")

class ScoringRequest(BaseModel):
    client_id:    Optional[str]  = Field(None, description="Identifiant client (généré si absent)")
    mobile_money: MobileMoney
    telco:        Telco
    profil:       ProfilClient
    with_explanation: bool       = Field(True,  description="Générer l'explication Claude AI")

class ScoringResponse(BaseModel):
    request_id:      str
    client_id:       str
    score:           int
    score_max:       int = 900
    proba:           float
    risk_level:      str
    risk_emoji:      str
    decision:        str
    montant_max_xof: int
    duree_max_mois:  int
    explanation:     Optional[str]
    recommendation:  Optional[str]
    shap_factors:    dict
    latency_ms:      int
    timestamp:       str

class BatchRequest(BaseModel):
    requests: list[ScoringRequest] = Field(..., min_items=1, max_items=50)

class BatchResponse(BaseModel):
    batch_id:    str
    total:       int
    success:     int
    failed:      int
    results:     list[dict]
    latency_ms:  int
    timestamp:   str


# ══════════════════════════════════════════════════════════════════════
# 4. CHARGEMENT MODÈLE (lazy, une seule fois)
# ══════════════════════════════════════════════════════════════════════

_model      = None
_explainer  = None
_feat_names = None

def load_artifacts():
    global _model, _explainer, _feat_names
    if _model is None:
        try:
            _model      = joblib.load(MODEL_DIR / "xgboost_model.pkl")
            _explainer  = joblib.load(MODEL_DIR / "shap_explainer.pkl")
            _feat_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Modèle non disponible : {e}")


# ══════════════════════════════════════════════════════════════════════
# 5. LOGIQUE DE SCORING
# ══════════════════════════════════════════════════════════════════════

def build_feature_vector(req: ScoringRequest) -> dict:
    mm, tel, pro = req.mobile_money, req.telco, req.profil
    montant_total = mm.nb_transactions_total * mm.montant_moyen_xof
    return {
        "nb_transactions_total":      mm.nb_transactions_total,
        "montant_moyen_xof":          mm.montant_moyen_xof,
        "montant_median_xof":         mm.montant_moyen_xof * 0.85,
        "montant_std_xof":            mm.montant_moyen_xof * 0.4,
        "montant_total_xof":          montant_total,
        "montant_max_xof":            mm.montant_moyen_xof * 3,
        "montant_min_xof":            mm.montant_moyen_xof * 0.1,
        "volatilite_montant":         0.4,
        "regularite_mensuelle":       mm.regularite_mensuelle,
        "ratio_epargne":              mm.ratio_epargne,
        "ratio_transferts":           0.45,
        "mois_actifs":                min(pro.anciennete_mois, 18),
        "tx_mois_moyen":              mm.nb_transactions_total / max(pro.anciennete_mois, 1),
        "recence_jours":              mm.recence_jours,
        "nb_tx_30j":                  mm.nb_tx_30j,
        "montant_sum_30j":            mm.montant_moyen_xof * mm.nb_tx_30j,
        "nb_tx_90j":                  mm.nb_tx_90j,
        "montant_sum_90j":            mm.montant_moyen_xof * mm.nb_tx_90j,
        "nb_tx_180j":                 mm.nb_transactions_total,
        "montant_sum_180j":           montant_total,
        "tendance_activite":          mm.tendance_activite,
        "heure_mediane":              13.0,
        "heure_std":                  2.5,
        "ratio_tx_heures_ouvrables":  0.75,
        "nb_recharges_6m":            tel.nb_recharges_6m,
        "regularite_recharge":        tel.regularite_recharge,
        "nb_contacts_uniques_30j":    tel.nb_contacts_uniques_30j,
        "duree_moy_appel_min":        tel.duree_moy_appel_min,
        "jours_actif_30j":            tel.jours_actif_30j,
        "a_internet_mobile":          tel.a_internet_mobile,
        "score_connectivite":         (tel.jours_actif_30j/30)*0.4 + tel.a_internet_mobile*0.2 + (tel.nb_contacts_uniques_30j/100)*0.4,
        "score_stabilite_telco":      tel.regularite_recharge*0.5 + (tel.nb_recharges_6m/30)*0.3 + (tel.duree_moy_appel_min/10)*0.2,
        "log_montant_recharge":       np.log1p(tel.montant_moy_recharge),
        "age":                        pro.age,
        "anciennete_mois":            pro.anciennete_mois,
        "log_anciennete":             np.log1p(pro.anciennete_mois),
        "a_emprunte_avant":           pro.a_emprunte_avant,
        "tier_urbain":                TIER_URBAIN.get(pro.region, 1),
        "score_operateur":            SCORE_OPERATEUR.get(pro.operateur, 1),
    }

def compute_score(features: dict) -> tuple[int, float, dict]:
    load_artifacts()
    X     = pd.DataFrame([features]).reindex(columns=_feat_names, fill_value=0)
    proba = float(_model.predict_proba(X)[:, 1][0])
    score = int(300 + proba * 600)

    sv     = _explainer.shap_values(X)[0]
    series = pd.Series(sv, index=_feat_names).sort_values(key=abs, ascending=False)
    factors = {
        feat: {
            "shap":   round(float(val), 4),
            "valeur": round(float(X[feat].iloc[0]), 2),
            "impact": "positif" if val > 0 else "négatif",
        }
        for feat, val in series.head(5).items()
    }
    return score, round(proba, 4), factors

def run_scoring(req: ScoringRequest) -> dict:
    t0       = time.time()
    features = build_feature_vector(req)
    score, proba, factors = compute_score(features)
    risk_level, risk_emoji, montant_max, duree_max, decision = _assess_risk(score)

    explanation    = None
    recommendation = None

    if req.with_explanation:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.agent.scoring_agent import node_generate_explanation, ScoringState
            state = ScoringState(
                client_id=req.client_id or "api_client",
                raw_features=features,
                score=score, proba=proba, shap_factors=factors,
                risk_level=risk_level, risk_emoji=risk_emoji,
                explanation="", recommendation="",
                decision=decision, montant_max_xof=montant_max,
                duree_max_mois=duree_max, error=None,
            )
            state = node_generate_explanation(state)
            explanation    = state["explanation"]
            recommendation = state["recommendation"]
        except Exception:
            explanation    = "Explication indisponible."
            recommendation = "Révision manuelle recommandée."

    return {
        "request_id":      str(uuid.uuid4())[:12],
        "client_id":       req.client_id or str(uuid.uuid4())[:12],
        "score":           score,
        "score_max":       900,
        "proba":           proba,
        "risk_level":      risk_level,
        "risk_emoji":      risk_emoji,
        "decision":        decision,
        "montant_max_xof": montant_max,
        "duree_max_mois":  duree_max,
        "explanation":     explanation,
        "recommendation":  recommendation,
        "shap_factors":    factors,
        "latency_ms":      int((time.time() - t0) * 1000),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════
# 6. ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Santé"])
def root():
    return {
        "service":  "CréditScore AI",
        "version":  "1.0.0",
        "status":   "running",
        "docs":     "/docs",
        "redoc":    "/redoc",
    }

@app.get("/health", tags=["Santé"])
def health():
    """Vérifie que le modèle est chargeable."""
    try:
        load_artifacts()
        return {"status": "healthy", "model": "loaded", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post(
    "/v1/score",
    response_model=ScoringResponse,
    tags=["Scoring"],
    summary="Scorer un client",
    description="Analyse les données Mobile Money + Telco et retourne un score crédit avec explication.",
)
def score_single(
    req: ScoringRequest,
    credentials: dict = Depends(verify_api_key),
):
    try:
        result = run_scoring(req)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.post(
    "/v1/score/batch",
    response_model=BatchResponse,
    tags=["Scoring"],
    summary="Scorer plusieurs clients (max 50)",
    description="Traitement en lot. Retourne les résultats dans l'ordre des requêtes.",
)
def score_batch(
    req: BatchRequest,
    credentials: dict = Depends(verify_api_key),
):
    t0      = time.time()
    results = []
    success = 0
    failed  = 0

    for r in req.requests:
        try:
            result = run_scoring(r)
            results.append({"status": "success", **result})
            success += 1
        except Exception as e:
            results.append({
                "status":    "error",
                "client_id": r.client_id,
                "error":     str(e),
            })
            failed += 1

    return {
        "batch_id":   str(uuid.uuid4())[:12],
        "total":      len(req.requests),
        "success":    success,
        "failed":     failed,
        "results":    results,
        "latency_ms": int((time.time() - t0) * 1000),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

@app.get(
    "/v1/model/info",
    tags=["Modèle"],
    summary="Informations sur le modèle",
)
def model_info(credentials: dict = Depends(verify_api_key)):
    """Retourne les métriques et features du modèle en production."""
    try:
        metrics_path = MODEL_DIR / "metrics.json"
        shap_path    = MODEL_DIR / "shap_importance.csv"
        load_artifacts()

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        top_features = []
        if shap_path.exists():
            df = pd.read_csv(shap_path, index_col=0)
            top_features = df.head(10).reset_index().rename(
                columns={"index":"feature","importance":"shap_importance"}
            ).to_dict("records")

        return {
            "model_type":    "XGBoost + CalibratedClassifierCV",
            "n_features":    len(_feat_names) if _feat_names else 0,
            "score_range":   [300, 900],
            "metrics":       metrics,
            "top_features":  top_features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Route non trouvée", "docs": "/docs"})

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Erreur interne du serveur"})


# ══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)