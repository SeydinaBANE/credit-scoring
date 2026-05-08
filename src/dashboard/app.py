"""
Dashboard Streamlit — CréditScore AI
Thème bleu/blanc · Sidebar native · Interface agent de crédit
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RISK_THRESHOLDS

# ── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CréditScore AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME BLEU/BLANC ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f4ff !important;
    color: #1e293b !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1d4ed8 0%, #1e3a8a 100%) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    color: white !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox svg { fill: white !important; }
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
    margin: 16px 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: white !important;
    color: #1d4ed8 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #eff6ff !important;
}

.main .block-container {
    padding: 28px 32px !important;
    max-width: 1200px !important;
}
.card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.card-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
}
.score-num {
    font-size: 80px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -3px;
}
.score-green  { color: #059669; }
.score-yellow { color: #d97706; }
.score-orange { color: #ea580c; }
.score-red    { color: #dc2626; }

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 16px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red    { background: #fee2e2; color: #991b1b; }

.shap-row { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.shap-label { width:200px; font-size:12px; color:#64748b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.shap-track { flex:1; height:8px; background:#f1f5f9; border-radius:99px; overflow:hidden; }
.shap-fill-pos { height:8px; background:linear-gradient(90deg,#3b82f6,#06b6d4); border-radius:99px; }
.shap-fill-neg { height:8px; background:linear-gradient(90deg,#f43f5e,#f97316); border-radius:99px; }
.shap-val { width:50px; text-align:right; font-size:11px; font-family:'IBM Plex Mono',monospace; color:#94a3b8; }

.explain-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #1e40af;
    margin-bottom: 12px;
}
.reco-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #92400e;
}
.profil-card {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 14px;
    margin-top: 10px;
    font-size: 13px;
    color: white;
    line-height: 2;
}
.mini-metric { text-align:center; padding:12px; background:#f8faff; border-radius:10px; }
.mini-metric-val { font-size:20px; font-weight:700; color:#1d4ed8; }
.mini-metric-label { font-size:11px; color:#94a3b8; margin-top:2px; font-family:'IBM Plex Mono',monospace; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar toujours visible - cacher le bouton collapse */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
section[data-testid="stSidebar"] { transform: none !important; min-width: 300px !important; width: 300px !important; }
section[data-testid="stSidebar"][aria-expanded="false"] { margin-left: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

DATA_DIR  = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

@st.cache_data
def load_clients():
    return pd.read_csv(DATA_DIR / "clients.csv")

@st.cache_data
def load_features():
    return pd.read_csv(DATA_DIR / "features.csv", index_col="client_id")

_COLOR_MAP = {"🟢": "green", "🟡": "yellow", "🟠": "orange", "🔴": "red"}

def score_color(s):
    for threshold, _, emoji, *_ in RISK_THRESHOLDS:
        if s >= threshold:
            return _COLOR_MAP[emoji]
    return "red"

def render_gauge(score):
    pct = int((score - 300) / 600 * 100)
    c   = {"green":"#059669","yellow":"#d97706","orange":"#ea580c","red":"#dc2626"}[score_color(score)]
    cls = f"score-{score_color(score)}"
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 10px;">
        <div class="score-num {cls}">{score}</div>
        <div style="color:#94a3b8;font-size:12px;font-family:'IBM Plex Mono',monospace;margin-top:2px;">/ 900 pts</div>
        <div style="margin-top:14px;background:#f1f5f9;border-radius:99px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;height:10px;background:{c};border-radius:99px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#cbd5e1;
                    margin-top:4px;font-family:'IBM Plex Mono',monospace;">
            <span>300</span><span>550</span><span>700</span><span>900</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_badge(decision):
    if "APPROUVÉ" in decision:
        cls, icon = "badge-green", "✓"
    elif "RÉVISION" in decision:
        cls, icon = "badge-yellow", "⚠"
    else:
        cls, icon = "badge-red", "✗"
    st.markdown(f'<span class="badge {cls}">{icon} {decision}</span>', unsafe_allow_html=True)

def render_shap(factors):
    if not factors: return
    mx = max(abs(v["shap"]) for v in factors.values()) or 1
    for feat, info in factors.items():
        pct  = int(abs(info["shap"]) / mx * 100)
        pos  = info["impact"] == "positif"
        fill = "shap-fill-pos" if pos else "shap-fill-neg"
        arrow = "↑" if pos else "↓"
        color = "#3b82f6" if pos else "#f43f5e"
        label = feat.replace("_", " ")
        st.markdown(f"""
        <div class="shap-row">
            <div class="shap-label" title="{feat}">{label}</div>
            <div class="shap-track">
                <div class="{fill}" style="width:{pct}%"></div>
            </div>
            <div class="shap-val" style="color:{color}">{arrow} {info['valeur']}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px;">
        <div style="font-size:22px;font-weight:700;color:white;letter-spacing:-0.5px;">💳 CréditScore AI</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.55);margin-top:3px;font-family:'IBM Plex Mono',monospace;">
            Scoring alternatif · Afrique de l'Ouest
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    mode = st.radio(
        "Mode de saisie",
        ["👤 Client existant", "✏️ Saisie manuelle"],
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "👤 Client existant":
        clients  = load_clients()
        features = load_features()

        selected = st.selectbox(
            "Sélectionner un client",
            clients["client_id"].tolist(),
            format_func=lambda x: f"🔹 {x[:16]}"
        )

        row = clients[clients["client_id"] == selected].iloc[0]
        st.markdown(f"""
        <div class="profil-card">
            <b>Profil client</b><br>
            Âge : {row['age']} ans<br>
            Région : {row['region']}<br>
            Opérateur : {row['operateur']}<br>
            Ancienneté : {row['anciennete_mois']} mois<br>
            Emprunté avant : {'✅ Oui' if row['a_emprunte_avant'] else '❌ Non'}
        </div>
        """, unsafe_allow_html=True)

        raw_features = features.loc[selected].to_dict() if selected in features.index else {}

    else:
        st.markdown('<p style="color:white;font-size:13px;font-weight:600;margin:0 0 4px;">Mobile Money</p>', unsafe_allow_html=True)
        nb_tx     = st.number_input("Nb transactions", 0, 500, 20)
        montant_m = st.number_input("Montant moyen (XOF)", 0, 500000, 15000, step=1000)
        reg_mens  = st.slider("Régularité mensuelle", 0.0, 1.0, 0.65, 0.05)

        st.markdown('<p style="color:white;font-size:13px;font-weight:600;margin:8px 0 4px;">Telco</p>', unsafe_allow_html=True)
        nb_rech   = st.number_input("Recharges (6 mois)", 0, 60, 12)
        reg_rech  = st.slider("Régularité recharges", 0.0, 1.0, 0.70, 0.05)
        jours_act = st.number_input("Jours actifs (30j)", 0, 30, 20)
        internet  = st.checkbox("Internet mobile", value=True)

        st.markdown('<p style="color:white;font-size:13px;font-weight:600;margin:8px 0 4px;">Profil</p>', unsafe_allow_html=True)
        age        = st.number_input("Âge", 18, 80, 32)
        anciennete = st.number_input("Ancienneté (mois)", 1, 120, 18)
        emprunte   = st.checkbox("A déjà emprunté")
        region_sel = st.selectbox("Région", ["Dakar","Thiès","Saint-Louis","Kaolack","Ziguinchor","Diourbel","Tambacounda","Kolda"])
        tier       = {"Dakar":3,"Thiès":2,"Saint-Louis":2,"Kaolack":1,"Ziguinchor":1,"Diourbel":1,"Tambacounda":0,"Kolda":0}

        selected = f"manuel_{age}_{region_sel}"
        raw_features = {
            "nb_transactions_total": nb_tx, "montant_moyen_xof": montant_m,
            "montant_median_xof": montant_m*0.85, "montant_std_xof": montant_m*0.4,
            "montant_total_xof": nb_tx*montant_m, "montant_max_xof": montant_m*3,
            "montant_min_xof": montant_m*0.1, "volatilite_montant": 0.4,
            "regularite_mensuelle": reg_mens, "ratio_epargne": 0.1,
            "ratio_transferts": 0.45, "mois_actifs": min(anciennete, 18),
            "tx_mois_moyen": nb_tx/max(anciennete,1), "recence_jours": 10,
            "nb_tx_30j": max(1,nb_tx//6), "montant_sum_30j": montant_m*max(1,nb_tx//6),
            "nb_tx_90j": max(1,nb_tx//2), "montant_sum_90j": montant_m*max(1,nb_tx//2),
            "nb_tx_180j": nb_tx, "montant_sum_180j": montant_m*nb_tx,
            "tendance_activite": 0.2, "heure_mediane": 13, "heure_std": 2.5,
            "ratio_tx_heures_ouvrables": 0.75, "nb_recharges_6m": nb_rech,
            "regularite_recharge": reg_rech, "nb_contacts_uniques_30j": 20,
            "duree_moy_appel_min": 3.5, "jours_actif_30j": jours_act,
            "a_internet_mobile": int(internet),
            "score_connectivite": (jours_act/30)*0.4+int(internet)*0.2+0.2,
            "score_stabilite_telco": reg_rech*0.5+(nb_rech/30)*0.3+0.2,
            "log_montant_recharge": np.log1p(montant_m*0.2),
            "age": age, "anciennete_mois": anciennete,
            "log_anciennete": np.log1p(anciennete),
            "a_emprunte_avant": int(emprunte),
            "tier_urbain": tier.get(region_sel,1), "score_operateur": 2,
        }

    st.divider()
    run_btn = st.button("⚡  Analyser le client", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:28px;">
    <div style="font-size:26px;font-weight:700;color:#0f172a;letter-spacing:-0.5px;">Tableau de bord scoring</div>
    <div style="font-size:13px;color:#64748b;margin-top:2px;font-family:'IBM Plex Mono',monospace;">
        Scoring alternatif · Mobile Money + Telco · Propulsé par Claude AI
    </div>
</div>
""", unsafe_allow_html=True)

if run_btn and raw_features:
    with st.spinner("Analyse en cours…"):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.agent.scoring_agent import score_client
            result = score_client(selected, raw_features)
            st.session_state.result = result
            st.session_state.history.insert(0, {
                "client_id": selected[:16],
                "score":     result["score"],
                "risque":    result["risk_level"],
                "decision":  result["decision"],
                "emoji":     result["risk_emoji"],
            })
            st.session_state.history = st.session_state.history[:8]
        except Exception as e:
            st.error(f"Erreur : {e}")

if st.session_state.result:
    r = st.session_state.result
    col1, col2 = st.columns([1, 1.7], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Score de crédit</div>', unsafe_allow_html=True)
        render_gauge(r["score"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Décision</div>', unsafe_allow_html=True)
        render_badge(r["decision"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="mini-metric" style="margin-top:14px;"><div class="mini-metric-val">{r["montant_max_xof"]//1000}k</div><div class="mini-metric-label">XOF max</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mini-metric" style="margin-top:14px;"><div class="mini-metric-val">{r["duree_max_mois"]}</div><div class="mini-metric-label">mois max</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:12px;text-align:center;font-size:12px;color:#94a3b8;font-family:\'IBM Plex Mono\',monospace;">Probabilité bon payeur : <b style="color:#1d4ed8">{r["proba"]*100:.1f}%</b></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Facteurs déterminants (SHAP)</div>', unsafe_allow_html=True)
        render_shap(r["shap_factors"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Analyse — Claude AI</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="explain-box">{r["explanation"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-label" style="margin-top:12px;">Recommandation agent</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="reco-box">{r["recommendation"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Historique des analyses</div>', unsafe_allow_html=True)
        hcols = st.columns([2,1,1.2,1.8])
        for lbl, col in zip(["CLIENT","SCORE","RISQUE","DÉCISION"], hcols):
            col.markdown(f'<div style="font-size:10px;font-weight:600;color:#94a3b8;letter-spacing:.08em;padding-bottom:8px;border-bottom:2px solid #f1f5f9;">{lbl}</div>', unsafe_allow_html=True)
        colors = {"faible":"#059669","modéré":"#d97706","élevé":"#ea580c","très élevé":"#dc2626"}
        for h in st.session_state.history:
            c = colors.get(h["risque"],"#64748b")
            hcols = st.columns([2,1,1.2,1.8])
            hcols[0].markdown(f'<div style="font-size:12px;color:#475569;font-family:\'IBM Plex Mono\',monospace;padding:8px 0;">{h["client_id"]}</div>', unsafe_allow_html=True)
            hcols[1].markdown(f'<div style="font-size:13px;font-weight:700;color:{c};padding:8px 0;">{h["score"]}</div>', unsafe_allow_html=True)
            hcols[2].markdown(f'<div style="font-size:12px;color:{c};padding:8px 0;">{h["emoji"]} {h["risque"]}</div>', unsafe_allow_html=True)
            hcols[3].markdown(f'<div style="font-size:11px;color:#64748b;font-family:\'IBM Plex Mono\',monospace;padding:8px 0;">{h["decision"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;background:white;border-radius:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="font-size:56px;margin-bottom:16px;">💳</div>
        <div style="font-size:18px;font-weight:600;color:#1e293b;">Prêt à analyser</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:8px;font-family:'IBM Plex Mono',monospace;">
            Sélectionnez un client dans la sidebar et cliquez sur Analyser
        </div>
        <div style="display:flex;justify-content:center;gap:32px;margin-top:32px;">
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#1d4ed8;">39</div>
                <div style="font-size:11px;color:#94a3b8;font-family:'IBM Plex Mono',monospace;">FEATURES</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#1d4ed8;">0.87</div>
                <div style="font-size:11px;color:#94a3b8;font-family:'IBM Plex Mono',monospace;">AUC-ROC</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#1d4ed8;">2k</div>
                <div style="font-size:11px;color:#94a3b8;font-family:'IBM Plex Mono',monospace;">CLIENTS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)