# 💳 CréditScore AI

> Scoring de crédit alternatif pour les **non-bankés en Afrique de l'Ouest**  
> Analyse les données **Mobile Money** (Wave, Orange Money) et **Telco** pour évaluer la solvabilité des clients sans historique bancaire.

---

## Aperçu

| Dashboard | API |
|-----------|-----|
| ![Dashboard](screnshots/dashbord.png) | ![API](screnshots/api.png) |

---

## Fonctionnement

```
Données Mobile Money + Telco
        ↓
Feature Engineering (39 features RFM, fenêtres temporelles, scores telco)
        ↓
XGBoost calibré → Score 300–900 + probabilité
        ↓
SHAP → Top 5 facteurs explicatifs
        ↓
Claude AI (via OpenRouter) → Explication en français + recommandation
        ↓
Décision : APPROUVÉ / RÉVISION MANUELLE / REFUSÉ
```

### Seuils de décision

| Score | Risque | Décision | Montant max | Durée max |
|-------|--------|----------|-------------|-----------|
| ≥ 700 | 🟢 Faible | APPROUVÉ | 500 000 XOF | 12 mois |
| 550–699 | 🟡 Modéré | RÉVISION MANUELLE | 200 000 XOF | 6 mois |
| 400–549 | 🟠 Élevé | RÉVISION MANUELLE | 75 000 XOF | 3 mois |
| < 400 | 🔴 Très élevé | REFUSÉ | — | — |

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle ML | XGBoost + CalibratedClassifierCV |
| Explicabilité | SHAP (TreeExplainer) |
| Agent IA | LangGraph + LangChain |
| LLM | Claude Haiku via OpenRouter |
| API | FastAPI + Pydantic v2 |
| Dashboard | Streamlit |
| Runtime | Python 3.11 + uv |

**Performance modèle** : AUC-ROC 0.86 · CV 0.873 ± 0.018

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/SeydinaBANE/credit-scoring.git
cd credit-scoring

# Installer les dépendances
uv sync

# Configurer les variables d'environnement
cp .env_exemple .env
# Éditer .env et ajouter votre clé OpenRouter
```

### Variables d'environnement (`.env`)

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_ID=anthropic/claude-haiku
```

---

## Lancement

```bash
# Dashboard Streamlit
uv run streamlit run src/dashboard/app.py

# API FastAPI (http://localhost:8000/docs)
uv run uvicorn src.api.main:app --reload --port 8000
```

---

## API

Toutes les routes nécessitent le header `X-API-Key`.

```bash
# Scorer un client
curl -X POST http://localhost:8000/v1/score \
  -H "X-API-Key: cs-dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "mobile_money": {
      "nb_transactions_total": 45,
      "montant_moyen_xof": 12000,
      "regularite_mensuelle": 0.75,
      "nb_tx_30j": 8,
      "nb_tx_90j": 22,
      "recence_jours": 3
    },
    "telco": {
      "nb_recharges_6m": 18,
      "regularite_recharge": 0.80,
      "jours_actif_30j": 24,
      "a_internet_mobile": 1,
      "montant_moy_recharge": 2000
    },
    "profil": {
      "age": 34,
      "anciennete_mois": 24,
      "region": "Dakar",
      "operateur": "Wave"
    }
  }'
```

**Réponse :**
```json
{
  "score": 724,
  "risk_level": "faible",
  "decision": "APPROUVÉ",
  "montant_max_xof": 500000,
  "duree_max_mois": 12,
  "explanation": "Le client présente un profil solide...",
  "recommendation": "Prêt recommandé sous conditions standard.",
  "shap_factors": { ... }
}
```

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | État du service |
| `POST` | `/v1/score` | Scorer un client |
| `POST` | `/v1/score/batch` | Scorer jusqu'à 50 clients |
| `GET` | `/v1/model/info` | Métriques et features du modèle |
| `GET` | `/docs` | Documentation Swagger |

---

## Régénérer les données et le modèle

```bash
# Générer des données synthétiques
uv run python data/generate_data.py

# Recalculer les features
uv run python src/features/feature_engeneering.py

# Réentraîner le modèle
uv run python src/model/train.py
```

---

## Structure

```
credit-scoring/
├── src/
│   ├── config.py              # Constantes partagées (seuils, mappings)
│   ├── agent/scoring_agent.py # Pipeline LangGraph (3 nœuds)
│   ├── api/main.py            # API FastAPI
│   ├── dashboard/app.py       # Interface Streamlit
│   ├── features/              # Feature engineering
│   └── model/train.py         # Entraînement XGBoost
├── data/
│   └── generate_data.py       # Générateur de données synthétiques
├── models/                    # Artifacts ML (non versionnés)
├── .env_exemple               # Template de configuration
└── pyproject.toml
```

---

## Contexte

Ce projet s'adresse aux institutions de microfinance et fintech actives en Afrique de l'Ouest souhaitant évaluer la solvabilité de clients sans compte bancaire. En l'absence d'historique de crédit traditionnel, le modèle exploite des signaux comportementaux issus du Mobile Money et des données télécom comme proxies de fiabilité financière.
