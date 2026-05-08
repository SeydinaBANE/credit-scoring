# CLAUDE.md

Ce fichier fournit des indications à Claude Code (claude.ai/code) pour travailler dans ce dépôt.

## Présentation du projet

**CréditScore AI** — SaaS de scoring de crédit alternatif pour les non-bankés en Afrique de l'Ouest. Analyse les données Mobile Money (Wave, Orange Money) et Telco pour produire un score crédit (300–900), une explication en français générée par Claude AI, et une décision d'octroi avec montant et durée suggérés.

## Commandes

Le projet utilise `uv` pour la gestion des dépendances (Python 3.11).

```bash
# Installer les dépendances
uv sync

# Lancer le dashboard Streamlit
uv run streamlit run src/dashboard/app.py

# Lancer le serveur FastAPI
uv run uvicorn src.api.main:app --reload --port 8000

# Reconstruire les features depuis les données brutes (clients.csv + transactions.csv + telco.csv)
uv run python src/features/feature_engeneering.py

# Réentraîner le modèle (nécessite features.csv + labels.csv)
uv run python src/model/train.py

# Tester l'agent LangGraph directement (exécute sur 2 clients exemples)
uv run python src/agent/scoring_agent.py
```

## Variables d'environnement

Copier `.env_exemple` vers `.env` :

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Obligatoire — appels LLM (Claude Haiku via OpenRouter) |
| `OPENROUTER_BASE_URL` | Défaut : `https://openrouter.ai/api/v1` |
| `MODEL_ID` | Défaut : `anthropic/claude-haiku` |
| `API_KEY_DEV` | Clé API dev FastAPI (défaut : `cs-dev-key-123`) |
| `API_KEY_PROD` | Clé API prod FastAPI (défaut : `cs-prod-key-456`) |

## Architecture

### Flux de données

```
data/clients.csv + transactions.csv + telco.csv
    → src/features/feature_engeneering.py
    → data/features.csv + data/labels.csv
    → src/model/train.py
    → models/xgboost_model.pkl + shap_explainer.pkl + feature_names.pkl
```

À l'inférence :
```
Entrée API/Dashboard → build_feature_vector() → XGBoost (39 features) → top-5 facteurs SHAP
    → agent LangGraph → explication Claude AI → ScoringResponse
```

### Rôle de chaque module

- **`src/features/feature_engeneering.py`** — Agrège les transactions Mobile Money brutes (RFM, stats fenêtrées, patterns temporels) et les données Telco en 39 features ML. À relancer pour régénérer `data/features.csv`.

- **`src/model/train.py`** — Entraîne XGBoost avec calibration isotonique (`CalibratedClassifierCV`), calcule l'importance SHAP globale, sauvegarde les artifacts dans `models/`. Le SHAP est calculé sur le modèle interne (`calibrated_classifiers_[0].estimator`).

- **`src/agent/scoring_agent.py`** — Pipeline LangGraph en 4 nœuds : `compute_score` → `assess_risk` → `generate_explanation` → `make_decision`. Le LLM (via `langchain_openai.ChatOpenAI` pointé sur OpenRouter) génère les explications en français à partir des facteurs SHAP. Point d'entrée : `score_client(client_id, features_dict)`.

- **`src/api/main.py`** — Application FastAPI avec authentification par clé API (header `X-API-Key`). Endpoints : `POST /v1/score` (unitaire), `POST /v1/score/batch` (jusqu'à 50), `GET /v1/model/info`. L'API reconstruit elle-même le vecteur de features depuis l'entrée structurée (ne lit pas `features.csv`) ; elle appelle `node_generate_explanation` directement plutôt que le graphe LangGraph complet.

- **`src/dashboard/app.py`** — Interface Streamlit. Deux modes : sélection d'un client existant depuis `data/clients.csv` (charge les features pré-calculées de `data/features.csv`) ou saisie manuelle (construit les features à la volée). Appelle `score_client()` du module agent.

### Seuils de score

| Score | Risque | Décision | Montant max | Durée max |
|---|---|---|---|---|
| ≥ 700 | faible 🟢 | APPROUVÉ | 500 000 XOF | 12 mois |
| 550–699 | modéré 🟡 | RÉVISION MANUELLE | 200 000 XOF | 6 mois |
| 400–549 | élevé 🟠 | RÉVISION MANUELLE | 75 000 XOF | 3 mois |
| < 400 | très élevé 🔴 | REFUSÉ | 0 | 0 |

### Notes de conception importantes

- Les artifacts du modèle sont chargés de façon paresseuse et mis en cache dans des globaux de module (`_model`, `_explainer`, `_feat_names`).
- Formule du score : `score = int(300 + proba * 600)` où `proba` est la probabilité calibrée d'être un bon payeur.
- Le script `data/generate_data.py` génère les données d'entraînement synthétiques. Le dossier `data/` contient 2 000 clients sénégalais synthétiques.
- Les clés API dans `VALID_API_KEYS` ont des valeurs par défaut codées en dur si les variables d'environnement ne sont pas définies — toujours les configurer via `.env` en production.
