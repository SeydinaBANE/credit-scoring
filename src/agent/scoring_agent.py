"""
Agent de scoring LangGraph — Scoring de crédit alternatif
Pipeline : collect_features → compute_score → assess_risk → generate_explanation → make_decision
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import assess_risk as _assess_risk

load_dotenv()

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# ══════════════════════════════════════════════════════════════════════
# 1. LLM via OpenRouter
# ══════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    model=os.getenv("MODEL_ID", "anthropic/claude-haiku"),
    max_tokens=600,
    temperature=0.3,
)

# ══════════════════════════════════════════════════════════════════════
# 2. ÉTAT PARTAGÉ
# ══════════════════════════════════════════════════════════════════════

class ScoringState(TypedDict):
    # Entrée
    client_id:        str
    raw_features:     dict

    # Scoring
    score:            int         # 300–900
    proba:            float       # probabilité brute
    shap_factors:     dict        # top 5 facteurs SHAP

    # Évaluation
    risk_level:       str         # faible / modéré / élevé
    risk_emoji:       str

    # LLM
    explanation:      str         # texte généré par Claude
    recommendation:   str         # conseils pour l'agent

    # Décision finale
    decision:         str         # approuvé / révision_manuelle / refusé
    montant_max_xof:  int         # montant max suggéré
    duree_max_mois:   int         # durée max suggérée

    # Meta
    error:            Optional[str]


# ══════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES ARTIFACTS (lazy)
# ══════════════════════════════════════════════════════════════════════

_model       = None
_explainer   = None
_feat_names  = None

def load_artifacts():
    global _model, _explainer, _feat_names
    if _model is None:
        _model      = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        _explainer  = joblib.load(MODEL_DIR / "shap_explainer.pkl")
        _feat_names = joblib.load(MODEL_DIR / "feature_names.pkl")


# ══════════════════════════════════════════════════════════════════════
# 4. NŒUDS DU GRAPHE
# ══════════════════════════════════════════════════════════════════════

def node_compute_score(state: ScoringState) -> ScoringState:
    """Calcule le score XGBoost + valeurs SHAP."""
    try:
        load_artifacts()

        X = pd.DataFrame([state["raw_features"]]).reindex(
            columns=_feat_names, fill_value=0
        )

        proba  = float(_model.predict_proba(X)[:, 1][0])
        score  = int(300 + proba * 600)
        shap_v = _explainer.shap_values(X)[0]

        series  = pd.Series(shap_v, index=_feat_names).sort_values(key=abs, ascending=False)
        factors = {
            feat: {
                "shap":   round(float(val), 4),
                "valeur": round(float(X[feat].iloc[0]), 2),
                "impact": "positif" if val > 0 else "négatif",
            }
            for feat, val in series.head(5).items()
        }

        state["proba"]        = round(proba, 4)
        state["score"]        = score
        state["shap_factors"] = factors
        state["error"]        = None

    except Exception as e:
        state["error"] = f"Erreur scoring : {e}"
        state["score"] = 0
        state["proba"] = 0.0
        state["shap_factors"] = {}

    return state


def node_assess_risk(state: ScoringState) -> ScoringState:
    """Classifie le niveau de risque et calcule les paramètres de crédit."""
    risk_level, emoji, montant_max, duree_max, decision = _assess_risk(state["score"])
    state["risk_level"]      = risk_level
    state["risk_emoji"]      = emoji
    state["montant_max_xof"] = montant_max
    state["duree_max_mois"]  = duree_max
    state["decision"]        = decision
    return state


def node_generate_explanation(state: ScoringState) -> ScoringState:
    """Claude génère une explication et une recommandation en français."""

    if state.get("error"):
        state["explanation"]    = "Impossible de générer une explication (erreur de scoring)."
        state["recommendation"] = "Vérifier les données du client."
        return state

    # Formatage des facteurs SHAP pour le prompt
    facteurs_str = "\n".join([
        f"  - {feat} : {info['impact']} (valeur={info['valeur']}, SHAP={info['shap']})"
        for feat, info in state["shap_factors"].items()
    ])

    system_prompt = """Tu es un analyste crédit senior spécialisé en microfinance et inclusion financière en Afrique de l'Ouest.
Tu analyses des profils de clients non-bankés à partir de données alternatives (Mobile Money, telco).
Tes explications sont claires, factuelles et directement utilisables par un agent de crédit terrain.
Tu écris toujours en français. Tu es concis mais précis."""

    user_prompt = f"""Analyse ce profil de crédit et rédige une explication professionnelle.

DONNÉES DU CLIENT :
- ID client     : {state['client_id']}
- Score crédit  : {state['score']}/900
- Niveau risque : {state['risk_level']} {state['risk_emoji']}
- Probabilité bon payeur : {state['proba']*100:.1f}%

FACTEURS DÉTERMINANTS (analyse SHAP) :
{facteurs_str}

PARAMÈTRES CRÉDIT SUGGÉRÉS :
- Montant max  : {state['montant_max_xof']:,} XOF
- Durée max    : {state['duree_max_mois']} mois

INSTRUCTIONS :
1. Rédige une explication en 3-4 phrases qui justifie le score en citant les 2-3 facteurs les plus importants.
   Traduis les noms techniques en langage compréhensible (ex: "regularite_recharge" → "régularité des recharges téléphoniques").
2. Puis, sur une nouvelle ligne après "RECOMMANDATION:", rédige 1-2 phrases de conseil pratique pour l'agent.

Format attendu :
EXPLICATION: [ton explication ici]
RECOMMANDATION: [ta recommandation ici]"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        content = response.content.strip()

        # Parse les deux sections
        expl_part  = ""
        reco_part  = ""

        if "EXPLICATION:" in content and "RECOMMANDATION:" in content:
            parts      = content.split("RECOMMANDATION:")
            expl_part  = parts[0].replace("EXPLICATION:", "").strip()
            reco_part  = parts[1].strip()
        else:
            expl_part = content
            reco_part = "Procéder selon les politiques internes."

        state["explanation"]    = expl_part
        state["recommendation"] = reco_part

    except Exception as e:
        state["explanation"]    = f"Erreur LLM : {e}"
        state["recommendation"] = "Révision manuelle recommandée."

    return state


# ══════════════════════════════════════════════════════════════════════
# 5. CONSTRUCTION DU GRAPHE
# ══════════════════════════════════════════════════════════════════════

def build_scoring_graph():
    graph = StateGraph(ScoringState)

    graph.add_node("compute_score",        node_compute_score)
    graph.add_node("assess_risk",          node_assess_risk)
    graph.add_node("generate_explanation", node_generate_explanation)

    graph.set_entry_point("compute_score")
    graph.add_edge("compute_score",        "assess_risk")
    graph.add_edge("assess_risk",          "generate_explanation")
    graph.add_edge("generate_explanation", END)

    return graph.compile()


# Instance globale
scoring_graph = build_scoring_graph()


# ══════════════════════════════════════════════════════════════════════
# 6. FONCTION PUBLIQUE (utilisée par l'API et le dashboard)
# ══════════════════════════════════════════════════════════════════════

def score_client(client_id: str, features: dict) -> dict:
    """
    Point d'entrée principal.
    Retourne le résultat complet du scoring pour un client.
    """
    initial_state: ScoringState = {
        "client_id":       client_id,
        "raw_features":    features,
        "score":           0,
        "proba":           0.0,
        "shap_factors":    {},
        "risk_level":      "",
        "risk_emoji":      "",
        "explanation":     "",
        "recommendation":  "",
        "decision":        "",
        "montant_max_xof": 0,
        "duree_max_mois":  0,
        "error":           None,
    }

    result = scoring_graph.invoke(initial_state)
    return result


# ══════════════════════════════════════════════════════════════════════
# 7. MAIN — TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Charger un vrai client depuis les features
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    X = pd.read_csv(DATA_DIR / "features.csv", index_col="client_id")

    # Tester sur 2 profils contrastés
    clients_test = {
        "bon_profil":    X.iloc[4].to_dict(),   # client index 4
        "profil_risque": X.iloc[7].to_dict(),   # client index 7
    }

    for label, features in clients_test.items():
        print(f"\n{'='*60}")
        print(f"TEST : {label}")
        print('='*60)

        result = score_client(label, features)

        print(f"Score      : {result['score']}/900")
        print(f"Risque     : {result['risk_emoji']} {result['risk_level']}")
        print(f"Décision   : {result['decision']}")
        print(f"Montant max: {result['montant_max_xof']:,} XOF")
        print(f"Durée max  : {result['duree_max_mois']} mois")
        print(f"\nEXPLICATION :")
        print(result['explanation'])
        print(f"\nRECOMMANDATION :")
        print(result['recommendation'])
        print(f"\nSHAP factors :")
        for feat, info in result['shap_factors'].items():
            print(f"  {feat:35s} {info['impact']:8s}  val={info['valeur']}")