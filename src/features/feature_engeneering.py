"""
Feature Engineering — Scoring de crédit alternatif
Agrège transactions Mobile Money + données telco en features ML
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import TIER_URBAIN, SCORE_OPERATEUR

REFERENCE_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURES MOBILE MONEY (RFM + ratios)
# ══════════════════════════════════════════════════════════════════════

def compute_mobile_money_features(df_tx: pd.DataFrame) -> pd.DataFrame:
    df = df_tx.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["mois"] = df["date"].dt.to_period("M")

    # ── RFM ────────────────────────────────────────────────────────
    recence = (
        df.groupby("client_id")["date"]
        .max()
        .apply(lambda d: (REFERENCE_DATE - d).days)
        .rename("recence_jours")
    )

    frequence = (
        df.groupby("client_id")
        .size()
        .rename("nb_transactions_total")
    )

    montant_total = (
        df.groupby("client_id")["montant_xof"]
        .sum()
        .rename("montant_total_xof")
    )

    # ── Statistiques montants ───────────────────────────────────────
    stats_montant = df.groupby("client_id")["montant_xof"].agg(
        montant_moyen_xof="mean",
        montant_median_xof="median",
        montant_std_xof="std",
        montant_max_xof="max",
        montant_min_xof="min",
    ).fillna(0)

    # Coefficient de variation (volatilité des montants)
    stats_montant["volatilite_montant"] = (
        stats_montant["montant_std_xof"] /
        stats_montant["montant_moyen_xof"].replace(0, np.nan)
    ).fillna(0).round(4)

    # ── Régularité mensuelle ────────────────────────────────────────
    tx_par_mois = (
        df.groupby(["client_id", "mois"])
        .size()
        .reset_index(name="nb_tx_mois")
    )
    regularite = tx_par_mois.groupby("client_id")["nb_tx_mois"].agg(
        mois_actifs="count",
        tx_mois_moyen="mean",
        tx_mois_std="std",
    ).fillna(0)
    regularite["regularite_mensuelle"] = (
        1 - (regularite["tx_mois_std"] /
             regularite["tx_mois_moyen"].replace(0, np.nan)).fillna(0)
    ).clip(0, 1).round(4)

    # ── Types de transactions ───────────────────────────────────────
    pivot_types = (
        df.groupby(["client_id", "type_tx"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_types.columns = [f"nb_{c.replace(' ', '_')}" for c in pivot_types.columns]

    # Ratio épargne (signal de discipline financière)
    if "nb_épargne" in pivot_types.columns:
        pivot_types["ratio_epargne"] = (
            pivot_types["nb_épargne"] / frequence
        ).fillna(0).round(4)
    else:
        pivot_types["ratio_epargne"] = 0.0

    # Ratio transferts reçus vs envoyés (signal réseau social)
    recus = pivot_types.get("nb_transfert_reçu", pd.Series(0, index=pivot_types.index))
    envoyes = pivot_types.get("nb_transfert_envoi", pd.Series(0, index=pivot_types.index))
    pivot_types["ratio_transferts"] = (
        recus / (envoyes + recus + 1e-9)
    ).round(4)

    # ── Fenêtres temporelles (30j / 90j / 180j) ────────────────────
    def tx_window(df, days):
        cutoff = REFERENCE_DATE - pd.Timedelta(days=days)
        sub = df[df["date"] >= cutoff]
        agg = sub.groupby("client_id").agg(
            nb_tx=("montant_xof", "count"),
            montant_sum=("montant_xof", "sum"),
        )
        agg.columns = [f"{c}_{days}j" for c in agg.columns]
        return agg

    w30  = tx_window(df, 30)
    w90  = tx_window(df, 90)
    w180 = tx_window(df, 180)

    # ── Tendance activité (croissance 90j vs 180j) ──────────────────
    tendance = w90[["nb_tx_90j"]].join(w180[["nb_tx_180j"]], how="outer").fillna(0)
    tendance["tendance_activite"] = (
        (tendance["nb_tx_90j"] * 2) /
        (tendance["nb_tx_180j"] + 1e-9) - 1
    ).clip(-1, 2).round(4)

    # ── Heure préférée (discipline horaire) ─────────────────────────
    heure_stats = df.groupby("client_id")["heure"].agg(
        heure_mediane="median",
        heure_std="std",
    ).fillna(0)
    # Transactions en heures ouvrables (8h-18h)
    df["heure_ouvrable"] = df["heure"].between(8, 18).astype(int)
    ratio_ouvrable = (
        df.groupby("client_id")["heure_ouvrable"]
        .mean()
        .rename("ratio_tx_heures_ouvrables")
        .round(4)
    )

    # ── Assemblage features Mobile Money ───────────────────────────
    mm_features = (
        recence
        .to_frame()
        .join(frequence)
        .join(montant_total)
        .join(stats_montant)
        .join(regularite[["mois_actifs", "tx_mois_moyen", "regularite_mensuelle"]])
        .join(pivot_types[["ratio_epargne", "ratio_transferts"]])
        .join(w30)
        .join(w90)
        .join(w180)
        .join(tendance[["tendance_activite"]])
        .join(heure_stats)
        .join(ratio_ouvrable)
        .fillna(0)
    )

    return mm_features


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES TELCO
# ══════════════════════════════════════════════════════════════════════

def compute_telco_features(df_telco: pd.DataFrame) -> pd.DataFrame:
    df = df_telco.set_index("client_id").copy()

    # Score de connectivité (réseau social proxy)
    df["score_connectivite"] = (
        (df["nb_contacts_uniques_30j"] / df["nb_contacts_uniques_30j"].max()) * 0.4 +
        (df["jours_actif_30j"] / 30) * 0.4 +
        df["a_internet_mobile"] * 0.2
    ).round(4)

    # Score de stabilité telco
    df["score_stabilite_telco"] = (
        df["regularite_recharge"].clip(0, 1) * 0.5 +
        (df["nb_recharges_6m"] / df["nb_recharges_6m"].max()) * 0.3 +
        (df["duree_moy_appel_min"] / df["duree_moy_appel_min"].max()) * 0.2
    ).round(4)

    # Montant recharge normalisé (log)
    df["log_montant_recharge"] = np.log1p(df["montant_moy_recharge"]).round(4)

    cols = [
        "nb_recharges_6m", "regularite_recharge", "nb_contacts_uniques_30j",
        "duree_moy_appel_min", "jours_actif_30j", "a_internet_mobile",
        "score_connectivite", "score_stabilite_telco", "log_montant_recharge",
    ]
    return df[cols]


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURES CLIENT (profil démographique)
# ══════════════════════════════════════════════════════════════════════

def compute_client_features(df_clients: pd.DataFrame) -> pd.DataFrame:
    df = df_clients.set_index("client_id").copy()

    df["tier_urbain"]    = df["region"].map(TIER_URBAIN).fillna(1)
    df["score_operateur"] = df["operateur"].map(SCORE_OPERATEUR).fillna(1)

    # Ancienneté log
    df["log_anciennete"] = np.log1p(df["anciennete_mois"]).round(4)

    cols = [
        "age", "anciennete_mois", "log_anciennete",
        "a_emprunte_avant", "tier_urbain", "score_operateur",
    ]
    return df[cols]


# ══════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    df_clients: pd.DataFrame,
    df_tx: pd.DataFrame,
    df_telco: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    print("Calcul features Mobile Money...")
    mm   = compute_mobile_money_features(df_tx)

    print("Calcul features Telco...")
    tel  = compute_telco_features(df_telco)

    print("Calcul features Client...")
    cli  = compute_client_features(df_clients)

    print("Assemblage de la matrice...")
    X = cli.join(mm, how="left").join(tel, how="left").fillna(0)
    y = df_clients.set_index("client_id")["bon_payeur"]

    print(f"\nMatrice features : {X.shape[0]} clients × {X.shape[1]} features")
    print(f"Features créées  : {list(X.columns)}")
    print(f"\nDistribution cible :")
    print(y.value_counts().to_string())

    return X, y


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Chargement des données...")
    clients = pd.read_csv(DATA_DIR / "clients.csv")
    tx      = pd.read_csv(DATA_DIR / "transactions.csv")
    telco   = pd.read_csv(DATA_DIR / "telco.csv")

    X, y = build_feature_matrix(clients, tx, telco)

    # Sauvegarde
    X.to_csv(DATA_DIR / "features.csv")
    y.to_csv(DATA_DIR / "labels.csv")
    print(f"\nSauvegardé → data/features.csv ({X.shape[1]} features)")
    print(f"Sauvegardé → data/labels.csv")

    # Aperçu
    print("\nAperçu features (5 premières lignes, 8 colonnes) :")
    print(X.iloc[:5, :8].round(2).to_string())