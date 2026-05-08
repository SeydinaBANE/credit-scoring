import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

np.random.seed(42)
random.seed(42)

N_CLIENTS = 2000
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 6, 30)

SENEGAL_REGIONS = [
    "Dakar", "Thiès", "Saint-Louis", "Ziguinchor",
    "Kaolack", "Diourbel", "Tambacounda", "Kolda"
]
OPERATEURS = ["Orange", "Wave", "Free", "Expresso"]
TYPES_TX = ["retrait", "dépôt", "transfert_envoi", "transfert_reçu",
            "paiement_marchand", "paiement_facture", "épargne"]

# ── 1. TABLE CLIENTS ────────────────────────────────────────────────
def generate_clients(n):
    profils = np.random.choice(
        ["stable_formel", "stable_informel", "irrégulier", "risqué"],
        n, p=[0.25, 0.35, 0.25, 0.15]
    )
    ages = np.where(
        profils == "stable_formel",  np.random.randint(28, 55, n),
        np.where(profils == "stable_informel", np.random.randint(22, 50, n),
        np.where(profils == "irrégulier", np.random.randint(18, 45, n),
                 np.random.randint(18, 40, n)))
    )
    return pd.DataFrame({
        "client_id":      [str(uuid.uuid4())[:12] for _ in range(n)],
        "age":            ages,
        "region":         np.random.choice(SENEGAL_REGIONS, n),
        "operateur":      np.random.choice(OPERATEURS, n, p=[0.40, 0.35, 0.15, 0.10]),
        "profil":         profils,
        "anciennete_mois": np.random.randint(3, 60, n),
        "a_emprunte_avant": np.random.choice([0, 1], n, p=[0.65, 0.35]),
        "defaut_historique": np.where(
            profils == "risqué", np.random.choice([0,1], n, p=[0.45, 0.55]),
            np.random.choice([0,1], n, p=[0.90, 0.10])
        ),
    })

clients = generate_clients(N_CLIENTS)

# ── 2. TABLE TRANSACTIONS MOBILE MONEY ──────────────────────────────
def montant_par_profil(profil):
    if profil == "stable_formel":
        return max(500, np.random.lognormal(10.5, 0.6))
    elif profil == "stable_informel":
        return max(200, np.random.lognormal(9.8, 0.8))
    elif profil == "irrégulier":
        return max(100, np.random.lognormal(9.2, 1.1))
    else:
        return max(100, np.random.lognormal(8.8, 1.3))

def nb_tx_par_profil(profil):
    mapping = {
        "stable_formel": (15, 50),
        "stable_informel": (8, 35),
        "irrégulier": (3, 20),
        "risqué": (1, 15),
    }
    lo, hi = mapping[profil]
    return np.random.randint(lo, hi)

transactions = []
for _, client in clients.iterrows():
    nb_tx = nb_tx_par_profil(client["profil"])
    for _ in range(nb_tx):
        delta = random.randint(0, (END_DATE - START_DATE).days)
        date_tx = START_DATE + timedelta(days=delta)
        heure   = np.random.choice(range(7, 22), p=[
            0.04,0.06,0.09,0.11,0.12,0.11,0.10,0.09,0.07,0.06,0.05,0.04,0.03,0.02,0.01
        ])
        transactions.append({
            "tx_id":       str(uuid.uuid4())[:12],
            "client_id":   client["client_id"],
            "date":        date_tx.strftime("%Y-%m-%d"),
            "heure":       heure,
            "type_tx":     np.random.choice(TYPES_TX),
            "montant_xof": round(montant_par_profil(client["profil"]), -1),
            "operateur":   client["operateur"],
            "region":      client["region"],
        })

df_tx = pd.DataFrame(transactions)

# ── 3. TABLE DONNÉES TELCO ──────────────────────────────────────────
def generate_telco(clients):
    rows = []
    for _, c in clients.iterrows():
        p = c["profil"]
        rows.append({
            "client_id":              c["client_id"],
            "nb_recharges_6m":        np.random.randint(
                *{"stable_formel":(12,30),"stable_informel":(8,25),
                  "irrégulier":(3,15),"risqué":(1,10)}[p]),
            "montant_moy_recharge":   round(
                {"stable_formel":3500,"stable_informel":2000,
                 "irrégulier":1200,"risqué":800}[p]
                * np.random.uniform(0.7, 1.3), -2),
            "regularite_recharge":    round(
                {"stable_formel":0.85,"stable_informel":0.70,
                 "irrégulier":0.45,"risqué":0.30}[p]
                * np.random.uniform(0.8, 1.2), 3),
            "nb_contacts_uniques_30j": np.random.randint(
                *{"stable_formel":(20,80),"stable_informel":(15,60),
                  "irrégulier":(5,35),"risqué":(3,25)}[p]),
            "duree_moy_appel_min":    round(np.random.uniform(
                *{"stable_formel":(2,8),"stable_informel":(1.5,7),
                  "irrégulier":(1,5),"risqué":(0.5,4)}[p]), 2),
            "jours_actif_30j":        np.random.randint(
                *{"stable_formel":(20,30),"stable_informel":(15,28),
                  "irrégulier":(8,20),"risqué":(3,15)}[p]),
            "a_internet_mobile":      np.random.choice(
                [1,0], p={"stable_formel":[0.85,0.15],
                          "stable_informel":[0.65,0.35],
                          "irrégulier":[0.40,0.60],
                          "risqué":[0.25,0.75]}[p]),
        })
    return pd.DataFrame(rows)

df_telco = generate_telco(clients)

# ── 4. TABLE LABELS (variable cible) ────────────────────────────────
def calcul_label(row):
    score = 0
    if row["profil"] == "stable_formel":    score += 3
    elif row["profil"] == "stable_informel": score += 2
    elif row["profil"] == "irrégulier":     score += 1
    if row["defaut_historique"] == 1:        score -= 3
    if row["anciennete_mois"] > 24:          score += 1
    if row["a_emprunte_avant"] == 1:         score += 1
    bruit = np.random.randint(-1, 2)
    return 1 if (score + bruit) >= 3 else 0

clients["bon_payeur"] = clients.apply(calcul_label, axis=1)

# ── 5. SAUVEGARDE ───────────────────────────────────────────────────
from pathlib import Path

DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

clients.to_csv(DATA_DIR / "clients.csv", index=False)
df_tx.to_csv(DATA_DIR / "transactions.csv", index=False)
df_telco.to_csv(DATA_DIR / "telco.csv", index=False)

print("=== GÉNÉRATION TERMINÉE ===")
print(f"Clients      : {len(clients):,}")
print(f"Transactions : {len(df_tx):,}")
print(f"Telco        : {len(df_telco):,}")
print(f"\nDistribution profils :")
print(clients["profil"].value_counts().to_string())
print(f"\nDistribution cible (bon_payeur) :")
print(clients["bon_payeur"].value_counts().to_string())
print(f"\nMontant moyen transaction : {df_tx['montant_xof'].mean():,.0f} XOF")
print(f"Plage dates              : {df_tx['date'].min()} → {df_tx['date'].max()}")
print(f"\nAperçu transactions :")
print(df_tx.head(3).to_string())