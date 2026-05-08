"""
Constantes partagées — source unique de vérité pour les seuils et mappings.
"""

# Encodage géographique (tier urbain) et opérateur
TIER_URBAIN = {
    "Dakar": 3, "Thiès": 2, "Saint-Louis": 2,
    "Kaolack": 1, "Ziguinchor": 1, "Diourbel": 1,
    "Tambacounda": 0, "Kolda": 0,
}

SCORE_OPERATEUR = {
    "Wave": 3, "Orange": 2, "Free": 1, "Expresso": 0,
}

# Politique de risque : (seuil_min, risk_level, emoji, montant_max_xof, duree_max_mois, decision)
RISK_THRESHOLDS = [
    (700, "faible",     "🟢", 500_000, 12, "APPROUVÉ"),
    (550, "modéré",     "🟡", 200_000,  6, "RÉVISION MANUELLE"),
    (400, "élevé",      "🟠",  75_000,  3, "RÉVISION MANUELLE"),
    (  0, "très élevé", "🔴",       0,  0, "REFUSÉ"),
]


def assess_risk(score: int) -> tuple[str, str, int, int, str]:
    """Retourne (risk_level, emoji, montant_max_xof, duree_max_mois, decision)."""
    for threshold, risk_level, emoji, montant_max, duree_max, decision in RISK_THRESHOLDS:
        if score >= threshold:
            return risk_level, emoji, montant_max, duree_max, decision
    return "très élevé", "🔴", 0, 0, "REFUSÉ"
