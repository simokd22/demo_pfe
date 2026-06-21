# -*- coding: utf-8 -*-
"""
Plateforme de recommandation éducative — Démo PFE
3 onglets : Recommandation (LTR) · Démarrage à froid (user/item) · Architecture & preuves

Lancement :
    streamlit run app.py

Modèles chargés depuis ./artefacts/ :
    - DKT (user cold-start)     : dkt_user.pt, course_mapping_user.csv, dkt_user_meta.json
    - EERNN-M (item cold-start) : eernn_item.pt, S1_embeddings.npy,
                                  course_mapping_item.csv, eernn_item_meta.json
    - GB-LTR (cas nominal)      : gb_ltr_D.pkl, gb_ltr_features.json, gb_ltr_test.csv
    - (optionnel) courses.csv pour les noms de cours
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import altair as alt

ART = Path(__file__).parent / "artefacts"

st.set_page_config(page_title="Recommandation éducative — PFE",
                   layout="wide", initial_sidebar_state="expanded")

# ======================================================================
# 0. IDENTITÉ VISUELLE (cohérente avec le thème Beamer de la soutenance)
# ======================================================================
BLUE, INK, ACCENT, MUTED = "#007ACC", "#0A2540", "#FF5722", "#5B6B7B"
GREY_BAR = "#9AA7B4"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1 {{ color:{INK}; font-weight:700; letter-spacing:-.02em; margin-bottom:.1rem; }}
h2, h3 {{ color:{INK}; }}

.eyebrow {{ font-size:.72rem; font-weight:700; letter-spacing:.14em;
            text-transform:uppercase; color:{BLUE}; margin:.1rem 0 .15rem; }}
.brand-rule {{ height:4px; width:118px; border-radius:2px;
               background:linear-gradient(90deg,{BLUE},{ACCENT}); margin:.35rem 0 .9rem; }}
.subtitle {{ color:{MUTED}; font-size:.97rem; margin-bottom:1.1rem; }}

/* cartes de métriques (signal bleu) */
div[data-testid="stMetric"] {{ background:#F5F8FB; border:1px solid #E3EAF2;
    border-left:4px solid {BLUE}; border-radius:10px; padding:14px 16px; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color:{INK}; font-weight:700; }}
div[data-testid="stMetric"] label {{ color:{MUTED}; }}

/* carte « résultat clé » (signal orange — un seul point fort par section) */
.hero-card {{ background:linear-gradient(135deg,#FFF3EE,#FFFFFF); border:1px solid #FFD9CC;
    border-left:4px solid {ACCENT}; border-radius:10px; padding:14px 16px; height:100%; }}
.hero-label {{ font-size:.78rem; color:{MUTED}; }}
.hero-value {{ font-size:1.9rem; font-weight:700; color:{ACCENT}; line-height:1.15; }}
.hero-sub {{ font-size:.74rem; color:{MUTED}; margin-top:.15rem; }}

/* onglets */
button[data-baseweb="tab"] {{ font-weight:600; font-size:.95rem; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{BLUE}; }}
[data-baseweb="tab-highlight"] {{ background:{BLUE}; }}

/* barre latérale */
.side-eyebrow {{ font-size:.66rem; font-weight:700; letter-spacing:.14em;
                 text-transform:uppercase; color:{BLUE}; }}
.side-title {{ font-size:1.02rem; font-weight:700; color:{INK}; line-height:1.25; margin-bottom:.1rem; }}
.side-rule {{ height:3px; width:64px; border-radius:2px;
              background:linear-gradient(90deg,{BLUE},{ACCENT}); margin:.3rem 0 .8rem; }}
.mod {{ display:flex; align-items:center; gap:.5rem; font-size:.9rem; color:{INK}; padding:.16rem 0; }}
.mod .dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; flex:0 0 auto; }}
.side-note {{ font-size:.8rem; color:{MUTED}; line-height:1.45; }}

/* nettoyage visuel pour la démo */

</style>
""", unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f"<div class='eyebrow'>{text}</div>", unsafe_allow_html=True)


def hero_metric(label, value, sub=""):
    st.markdown(
        f"<div class='hero-card'><div class='hero-label'>{label}</div>"
        f"<div class='hero-value'>{value}</div>"
        f"<div class='hero-sub'>{sub}</div></div>",
        unsafe_allow_html=True)


# ----- graphiques à barres horizontales, charte du projet -----
def _hbar(data, cat, val, label_col, color=BLUE, scheme=None, height=None):
    h = height or max(120, 38 * len(data))
    base = alt.Chart(data)
    x = alt.X(f"{val}:Q", axis=None, title=None,
              scale=alt.Scale(domain=[0, float(data[val].max()) * 1.18]))
    y = alt.Y(f"{cat}:N", sort="-x", title=None,
              axis=alt.Axis(labelColor=INK, labelFontSize=13, labelLimit=440,
                            ticks=False, domain=False))
    col = (alt.Color(f"{val}:Q", scale=alt.Scale(scheme=scheme), legend=None)
           if scheme else alt.value(color))
    bars = base.mark_bar(height=20, cornerRadiusTopRight=5,
                         cornerRadiusBottomRight=5).encode(x=x, y=y, color=col)
    txt = base.mark_text(align="left", baseline="middle", dx=6, color=INK,
                         fontSize=12, fontWeight=600).encode(
        x=x, y=y, text=alt.Text(f"{label_col}:N"))
    return (bars + txt).properties(height=h).configure_view(
        stroke=None).configure_axis(grid=False)


def _hbar_colored(data, cat, val, label_col, height=None):
    """Couleurs explicites par catégorie (colonne 'color')."""
    h = height or max(110, 46 * len(data))
    base = alt.Chart(data)
    x = alt.X(f"{val}:Q", axis=None, title=None,
              scale=alt.Scale(domain=[0, float(data[val].max()) * 1.18]))
    y = alt.Y(f"{cat}:N", sort="-x", title=None,
              axis=alt.Axis(labelColor=INK, labelFontSize=13, labelLimit=440,
                            ticks=False, domain=False))
    bars = base.mark_bar(height=24, cornerRadiusTopRight=5,
                         cornerRadiusBottomRight=5).encode(
        x=x, y=y, color=alt.Color("color:N", scale=None, legend=None))
    txt = base.mark_text(align="left", baseline="middle", dx=6, color=INK,
                         fontSize=12, fontWeight=600).encode(
        x=x, y=y, text=alt.Text(f"{label_col}:N"))
    return (bars + txt).properties(height=h).configure_view(
        stroke=None).configure_axis(grid=False)


# ======================================================================
# 1. DÉFINITIONS DES MODÈLES (identiques aux notebooks)
# ======================================================================
class DKT(nn.Module):
    """User cold-start — embedding par identifiant de cours."""
    def __init__(self, n_courses, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(2 * n_courses + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_courses)

    def forward(self, x, q):
        h, _ = self.lstm(self.embed(x))
        return torch.gather(self.out(h), 2, q.unsqueeze(-1)).squeeze(-1)


class EERNN_M(nn.Module):
    """Item cold-start — contenu sémantique (all-mpnet-base-v2)."""
    def __init__(self, sem_matrix, embed_dim=100, hidden_dim=100):
        super().__init__()
        self.sem_dim = sem_matrix.shape[1]
        self.register_buffer("S", torch.tensor(sem_matrix, dtype=torch.float32))
        self.proj = nn.Linear(self.sem_dim, embed_dim)
        self.lstm = nn.LSTM(2 * embed_dim, hidden_dim, batch_first=True)
        self.W1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)

    def item_emb(self, idx):
        return self.proj(self.S[idx])

    def forward(self, x, r, q):
        e = self.item_emb(x)
        r_onehot = (r == 1).float().unsqueeze(-1)
        xe = torch.cat([e * r_onehot, e * (1 - r_onehot)], dim=-1)
        h, _ = self.lstm(xe)
        e_q = self.item_emb(q)
        combined = torch.cat([h, e_q], dim=-1)
        y = torch.relu(self.W1(combined))
        return self.W2(y).squeeze(-1)


# ======================================================================
# 2. CHARGEURS D'ARTEFACTS (mis en cache, tolérants aux fichiers manquants)
# ======================================================================
def _exists(*names):
    return all((ART / n).exists() for n in names)


@st.cache_data
def load_course_table():
    """Renvoie un DataFrame course_idx, course_id, name (name optionnel)."""
    path = None
    for n in ("course_mapping_item.csv", "course_mapping_user.csv", "course_mapping.csv"):
        if (ART / n).exists():
            path = ART / n
            break
    if path is None:
        return None
    t = pd.read_csv(path)
    if "course_name" in t.columns:
        t = t.rename(columns={"course_name": "name"})
    elif "name" not in t.columns and (ART / "courses.csv").exists():
        c = pd.read_csv(ART / "courses.csv")[["course_id", "name"]]
        t = t.merge(c, on="course_id", how="left")
    if "name" not in t.columns:
        t["name"] = t["course_id"].astype(str)
    t["name"] = t["name"].fillna(t["course_id"].astype(str))
    return t


@st.cache_resource
def load_dkt():
    if not _exists("dkt_user.pt", "dkt_user_meta.json"):
        return None, None
    meta = json.load(open(ART / "dkt_user_meta.json"))
    model = DKT(meta["N_COURSES"], meta["EMBED_DIM"], meta["HIDDEN_DIM"])
    model.load_state_dict(torch.load(ART / "dkt_user.pt", map_location="cpu"))
    model.eval()
    return model, meta


@st.cache_resource
def load_eernn():
    if not _exists("eernn_item.pt", "S1_embeddings.npy", "eernn_item_meta.json"):
        return None, None, None
    meta = json.load(open(ART / "eernn_item_meta.json"))
    S1 = np.load(ART / "S1_embeddings.npy")
    model = EERNN_M(S1, meta["EMBED_DIM"], meta["HIDDEN_DIM"])
    model.load_state_dict(torch.load(ART / "eernn_item.pt", map_location="cpu"))
    model.eval()
    return model, meta, S1


@st.cache_data
def load_course_stats():
    """course_idx -> base_success_rate, n_interactions (optionnel)."""
    if not (ART / "course_stats.csv").exists():
        return None
    return pd.read_csv(ART / "course_stats.csv")


@st.cache_data
def load_s1():
    """Matrice d'embeddings sémantiques (caractérisation de profil — affichage seul)."""
    if not (ART / "S1_embeddings.npy").exists():
        return None
    return np.load(ART / "S1_embeddings.npy")


@st.cache_resource
def load_ltr():
    if not _exists("gb_ltr_D.pkl", "gb_ltr_features.json"):
        return None, None, None
    import joblib
    model = joblib.load(ART / "gb_ltr_D.pkl")
    features = json.load(open(ART / "gb_ltr_features.json"))
    test = pd.read_csv(ART / "gb_ltr_test.csv") if (ART / "gb_ltr_test.csv").exists() else None
    return model, features, test


@st.cache_data
def load_ltr_families():
    """{famille: [colonnes]} pour les 7 familles de signaux (optionnel)."""
    p = ART / "ltr_feature_families.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_ltr_importance():
    """{feature: importance} depuis gb_D (optionnel)."""
    p = ART / "ltr_importance.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


# ======================================================================
# 3. LOGIQUE D'INFÉRENCE
# ======================================================================
def dkt_recommend_gain(model, meta, history, topn=10):
    """Recommande sur tout le catalogue par GAIN MARGINAL :
    proba(cours | profil de l'apprenant) − proba(cours | aucun historique).
    Le gain annule le biais de base (commun à tous) et fait ressortir ce qui
    est spécifique à cet apprenant. Renvoie [(idx, proba_avec, gain)]."""
    N, MAXSEQ = meta["N_COURSES"], meta["MAX_SEQ"]
    x = np.zeros(MAXSEQ, dtype=np.int64)
    L = min(len(history), MAXSEQ)
    for t in range(L):
        cidx, corr = history[t]
        x[t] = cidx + corr * N + 1
    x0 = np.zeros(MAXSEQ, dtype=np.int64)
    with torch.no_grad():
        h_w, _ = model.lstm(model.embed(torch.tensor(x).unsqueeze(0)))
        probs_with = torch.sigmoid(model.out(h_w)[0, max(L - 1, 0)]).numpy()
        h_b, _ = model.lstm(model.embed(torch.tensor(x0).unsqueeze(0)))
        probs_base = torch.sigmoid(model.out(h_b)[0, 0]).numpy()
    gain = probs_with - probs_base
    seen = {c for c, _ in history}
    ranked = sorted(((i, float(probs_with[i]), float(gain[i]))
                     for i in range(N) if i not in seen),
                    key=lambda z: -z[2])
    return ranked[:topn]


def eernn_rank(model, meta, hist_idx, new_idx_list):
    """Score plusieurs cours nouveaux pour un même historique (réponses=succès).
    Renvoie un array de probabilités aligné sur new_idx_list."""
    MAXSEQ = meta["MAX_SEQ"]
    L = min(len(hist_idx), MAXSEQ)
    K = len(new_idx_list)
    x = np.zeros((K, MAXSEQ), dtype=np.int64)
    r = np.zeros((K, MAXSEQ), dtype=np.int64)
    q = np.zeros((K, MAXSEQ), dtype=np.int64)
    for t in range(L):
        x[:, t] = int(hist_idx[t])
        r[:, t] = 1
    for k, c in enumerate(new_idx_list):
        q[k, L - 1] = int(c)
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(x), torch.tensor(r), torch.tensor(q)))
    return pred[:, L - 1].numpy()


def thematic_relevance(S1, hist_idx, new_idx):
    """Similarité cosinus entre le profil de l'apprenant (moyenne des embeddings
    de son historique) et chaque cours candidat. Signal de contenu pur."""
    prof = S1[hist_idx].mean(axis=0)
    prof = prof / (np.linalg.norm(prof) + 1e-9)
    cand = S1[new_idx]
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
    return cand @ prof


def _minmax(a):
    a = np.asarray(a, dtype=float)
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 1e-9 else np.full_like(a, 0.5)


# ======================================================================
# 4. BARRE LATÉRALE
# ======================================================================
st.sidebar.markdown("<div class='side-eyebrow'>Projet de fin d'études</div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("<div class='side-title'>Recommandation éducative<br>"
                    "context-aware &amp; scalable</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='side-rule'></div>", unsafe_allow_html=True)

status = {
    "DKT — apprenant nouveau": _exists("dkt_user.pt", "dkt_user_meta.json"),
    "EERNN-M — cours nouveau": _exists("eernn_item.pt", "S1_embeddings.npy"),
    "GB-LTR — cas nominal": _exists("gb_ltr_D.pkl", "gb_ltr_features.json"),
    "Catalogue de cours": load_course_table() is not None,
}
st.sidebar.markdown("<div class='side-eyebrow'>Modules</div>", unsafe_allow_html=True)
for label, ok in status.items():
    dot = BLUE if ok else "#C9D3DE"
    st.sidebar.markdown(
        f"<div class='mod'><span class='dot' style='background:{dot}'></span>{label}</div>",
        unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-rule'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='side-note'>Les trois onglets répondent aux trois verrous du projet : "
    "<b>sparsité</b> (recommandation nominale), <b>démarrage à froid</b> "
    "(apprenant &amp; cours nouveaux), <b>passage à l'échelle</b> (pipeline Big Data).</div>",
    unsafe_allow_html=True)


# ======================================================================
# 5. EN-TÊTE + ONGLETS
# ======================================================================
eyebrow("Démonstration")
st.title("Plateforme de recommandation éducative")
st.markdown("<div class='brand-rule'></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Context-aware · Big Data · IA générative — "
            "atténuation de la sparsité, du démarrage à froid et passage à l'échelle</div>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "1 · Recommandation (cas nominal)",
    "2 · Démarrage à froid",
    "3 · Architecture & preuves",
])


# ----------------------------------------------------------------------
# ONGLET 1 — Learning-to-Rank
# ----------------------------------------------------------------------
with tab1:
    eyebrow("Sparsité · Learning-to-Rank")
    st.header("Recommandation pour un apprenant avec historique")
    model, features, test = load_ltr()
    families = load_ltr_families()
    importance = load_ltr_importance()
    if model is None:
        st.info("Module Learning-to-Rank indisponible.")
    elif test is None:
        st.info("Jeu de démonstration indisponible pour ce module.")
    else:
        # ===== Niveau 1 — Sur quels signaux le modèle s'appuie (7 familles) =====
        st.subheader("Sur quels signaux le modèle s'appuie")
        if families and importance:
            fam_imp = {fam: sum(importance.get(c, 0.0) for c in cols)
                       for fam, cols in families.items()}
            tot = sum(fam_imp.values()) or 1.0
            fam_df = pd.DataFrame({
                "Famille": list(fam_imp.keys()),
                "Importance": [100 * v / tot for v in fam_imp.values()],
            })
            fam_df["label"] = fam_df["Importance"].map(lambda v: f"{v:.1f} %")
            st.altair_chart(
                _hbar(fam_df, "Famille", "Importance", "label", scheme="blues"),
                use_container_width=True)
            st.caption("Importance relative des 7 familles de signaux (Gradient Boosting, "
                       "Config D). Lecture clé : l'**historique de scores domine** (≈ 68 %), "
                       "puis les clics VLE ; le **sentiment ne pèse que ≈ 2 %** et son retrait "
                       "ne coûte que +0,0002 d'AUC — illustration directe du caractère **dérivé** "
                       "du sentiment (information déjà portée par le score).")
        else:
            st.caption("Importance par famille de signaux indisponible.")

        st.divider()

        # ===== Niveau 2 & 3 — Apprenant : profil par famille + classement =====
        st.subheader("Apprenant : profil et recommandations")
        counts = test.groupby("id_student").size().sort_values(ascending=False)
        options = counts.index.tolist()
        labels = {sid_: f"{sid_}  ({n} évaluations)" for sid_, n in counts.items()}
        st.caption(f"{len(options)} apprenants disponibles — triés par nombre d'évaluations "
                   "(les plus riches d'abord, idéals pour la démonstration).")
        sid = st.selectbox("Apprenant", options, format_func=lambda s: labels.get(s, str(s)),
                           key="ltr_student")
        rows = test[test["id_student"] == sid].copy()
        X = rows[features].values
        rows["score"] = model.predict_proba(X)[:, 1]
        rows = rows.sort_values("score", ascending=False)

        # Profil de l'apprenant, groupé par famille de signaux
        if families:
            st.markdown("##### Profil de l'apprenant (par famille de signaux)")
            prof = rows.iloc[0]
            cols_disp = st.columns(min(len(families), 4))
            for k, (fam, cols) in enumerate(families.items()):
                present = [c for c in cols if c in rows.columns]
                if not present:
                    continue
                vals = prof[present].astype(float)
                with cols_disp[k % len(cols_disp)].expander(f"{fam} ({len(present)})"):
                    st.dataframe(pd.DataFrame({"valeur": vals.round(3)}),
                                 use_container_width=True)
            st.caption("Valeurs réelles de l'apprenant sur chacune des 7 familles : "
                       "le jury voit qui est cet apprenant à travers 7 angles de données.")

        # Classement des évaluations
        st.markdown(f"##### {len(rows)} évaluations classées pour l'apprenant {sid}")
        ranked = (rows[["id_assessment", "score", "label"]]
                  .assign(Rang=range(1, len(rows) + 1))
                  .rename(columns={"id_assessment": "Évaluation",
                                   "score": "Score prédit", "label": "Réel"})
                  [["Rang", "Évaluation", "Score prédit", "Réel"]])
        st.dataframe(
            ranked, use_container_width=True, hide_index=True,
            column_config={
                "Score prédit": st.column_config.ProgressColumn(
                    "Score prédit", format="%.3f", min_value=0.0, max_value=1.0),
            })
        st.caption("Cas nominal — apprenant avec historique riche : le LTR ordonne ses "
                   "évaluations par probabilité de réussite. Métriques validées "
                   "(test warm, per-user chronologique) : **AUC 0,8818 · NDCG@10 0,9829 · "
                   "F1 0,9430**, confirmées en validation croisée **CV-AUC 0,8630 ± 0,0036**.")


# ----------------------------------------------------------------------
# ONGLET 2 — Cold-start (user fonctionnel + item)
# ----------------------------------------------------------------------
with tab2:
    eyebrow("Démarrage à froid · transfert et stratification")
    st.header("Démarrage à froid")
    courses = load_course_table()

    sub_user, sub_item = st.tabs(["Apprenant nouveau (DKT)", "Cours nouveau (EERNN-M)"])

    # --- USER COLD-START : contribution méthodologique (vedette) + live ---
    with sub_user:
        model, meta = load_dkt()
        if model is None or courses is None:
            st.info("Module apprenant nouveau (DKT) indisponible.")
        else:
            # ===== 1. Contribution : fiabiliser l'évaluation =====
            st.subheader("Contribution — fiabiliser l'évaluation du cold-start utilisateur")
            st.write("Le protocole B&W naïf produisait des AUC **dégénérées** (0,5 ou 1,0 "
                     "selon le tirage) à cause du déséquilibre des classes (94 % de réussites). "
                     "Une **stratification** des sets — 30 utilisateurs avec négatifs + 20 "
                     "purement positifs — a rendu la mesure stable et fiable.")
            auc_mean = float(meta.get("AUC_moyenne_5sets", 0.9662))
            auc_std = float(meta.get("AUC_ecart_type", 0.0064))
            naive_std = 0.1972
            ratio = naive_std / auc_std if auc_std > 0 else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("Écart-type — naïf", f"{naive_std:.4f}",
                      help="AUC instable selon le tirage des sets")
            c2.metric("Écart-type — stratifié", f"{auc_std:.4f}",
                      help="Mesure devenue stable")
            with c3:
                hero_metric("Variance réduite", f"÷{ratio:.0f}",
                            "stratification du protocole B&W")

            std_df = pd.DataFrame({
                "Protocole": ["Protocole naïf", "Protocole stratifié"],
                "Écart-type": [naive_std, auc_std],
                "label": [f"{naive_std:.4f}", f"{auc_std:.4f}"],
                "color": [ACCENT, BLUE],
            })
            st.altair_chart(_hbar_colored(std_df, "Protocole", "Écart-type", "label"),
                            use_container_width=True)
            st.success(f"Aboutissement : **DKT — AUC = {auc_mean:.4f} ± {auc_std:.4f}** "
                       f"(protocole B&W étendu, stratifié, 5 sets de 50 apprenants nouveaux).")
            st.caption("L'AUC élevée n'est pas un score isolé : c'est le résultat d'un protocole "
                       "d'évaluation diagnostiqué puis corrigé. Le déséquilibre a été identifié, "
                       "pas subi.")

            st.divider()

            # ===== 2. Démonstration live : recommandation par gain marginal =====
            st.subheader("Recommandation — top-N sur le catalogue, par gain marginal")
            st.write("Un apprenant nouveau a **suivi avec succès** quelques cours. "
                     "La probabilité absolue de réussite est **saturée** (≈ 1 pour presque "
                     "tout, dataset à 94 % de succès) : la trier directement n'a pas de sens. "
                     "On recommande donc par **gain marginal** = proba(cours | profil de "
                     "l'apprenant) − proba(cours | aucun historique). Le gain annule le biais "
                     "de base et fait ressortir ce qui est **spécifique à cet apprenant**.")
            name2idx = dict(zip(courses["name"], courses["course_idx"]))
            picked = st.multiselect("Cours déjà suivis avec succès par l'apprenant",
                                    options=list(name2idx.keys()), key="dkt_hist")
            history = [(int(name2idx[nm]), 1) for nm in picked]

            if history:
                st.markdown("##### Profil de l'apprenant")
                prof_df = pd.DataFrame(
                    [(nm, "✓ Suivi avec succès") for nm in picked],
                    columns=["Cours suivi", "Statut"])
                st.table(prof_df)

                S1 = load_s1()
                if S1 is not None:
                    idx2name = dict(zip(courses["course_idx"], courses["name"]))
                    hist_idx = [i for i, _ in history]
                    pv = S1[hist_idx].mean(axis=0)
                    pv = pv / (np.linalg.norm(pv) + 1e-9)
                    cat = S1 / (np.linalg.norm(S1, axis=1, keepdims=True) + 1e-9)
                    sims = cat @ pv
                    seen = set(hist_idx)
                    near = sorted(((i, sims[i]) for i in range(len(sims)) if i not in seen),
                                  key=lambda z: -z[1])[:3]
                    themes = " · ".join(str(idx2name.get(i, i)) for i, _ in near)
                    st.info(f"**Thème dominant du profil** (par contenu) : proche de « {themes} ». "
                            f"Les recommandations ci-dessous, elles, viennent des **parcours "
                            f"d'apprenants similaires** (collaboratif) — elles peuvent donc "
                            f"diverger du thème, et c'est attendu.")

            topn = st.slider("Nombre de recommandations", 5, 20, 10, key="dkt_topn")
            if st.button("Recommander", key="dkt_go", disabled=len(history) == 0):
                idx2name = dict(zip(courses["course_idx"], courses["name"]))
                recs = dkt_recommend_gain(model, meta, history, topn=topn)
                out = pd.DataFrame(
                    [(r + 1, idx2name.get(i, i), round(g, 4), round(p, 3))
                     for r, (i, p, g) in enumerate(recs)],
                    columns=["Rang", "Cours recommandé", "Gain marginal", "Réussite prédite"])
                st.dataframe(
                    out, use_container_width=True, hide_index=True,
                    column_config={
                        "Gain marginal": st.column_config.NumberColumn(
                            "Gain marginal", format="%.4f"),
                        "Réussite prédite": st.column_config.ProgressColumn(
                            "Réussite prédite", format="%.3f",
                            min_value=0.0, max_value=1.0),
                    })
                st.caption("Classement par **gain marginal** (apport spécifique du profil) ; "
                           "la « Réussite prédite » reste saturée, d'où le tri par gain. "
                           "**Limite assumée** : conformément à la littérature (Bhattacharjee & "
                           "Wayllace 2025), le DKT est validé pour la **prédiction de réussite** "
                           "cold-start (AUC 0,9662), pas comme moteur de recommandation "
                           "personnalisée. Le catalogue, dominé par la tech, induit un biais de "
                           "popularité que ce classement ne corrige pas — la recommandation par "
                           "pertinence relève de modèles dédiés (DeepFM, IHGNN) ou du contenu "
                           "(EERNN, volet item).")

    # --- ITEM COLD-START : AUC validée (coeur) + illustration qualitative ---
    with sub_item:
        model, meta, S1 = load_eernn()
        stats = load_course_stats()
        if model is None or courses is None:
            st.info("Module cours nouveau (EERNN-M) indisponible.")
        else:
            # ===== 1. Le résultat validé : apport du contenu sémantique =====
            st.subheader("Résultat validé — apport du contenu sémantique")
            eernn_auc = float(meta.get("AUC_moyenne_5folds", 0.9589))
            eernn_std = float(meta.get("AUC_ecart_type", 0.0016))
            lstm_auc, lstm_std = 0.8353, 0.0088
            gain = eernn_auc - lstm_auc

            c1, c2, c3 = st.columns(3)
            c1.metric("LSTM-M — sans contenu", f"{lstm_auc:.4f}",
                      help=f"± {lstm_std:.4f} · baseline par identifiant de cours")
            c2.metric("EERNN-M — avec contenu", f"{eernn_auc:.4f}",
                      help=f"± {eernn_std:.4f} · contenu sémantique all-mpnet-base-v2")
            with c3:
                hero_metric("Gain de contenu", f"+{gain:.3f}",
                            "AUC attribuable au seul contenu sémantique")

            auc_df = pd.DataFrame({
                "Modèle": ["LSTM-M (sans contenu)", "EERNN-M (avec contenu)"],
                "AUC": [lstm_auc, eernn_auc],
                "label": [f"{lstm_auc:.4f}", f"{eernn_auc:.4f}"],
                "color": [GREY_BAR, BLUE],
            })
            st.altair_chart(_hbar_colored(auc_df, "Modèle", "AUC", "label"),
                            use_container_width=True)
            st.caption("AUC mesurée en cross-validation **cours-exclus** (vrai cold-start item). "
                       "L'AUC évalue le **classement** des réussites/échecs et reste fiable malgré "
                       "le déséquilibre des classes. Les deux modèles subissent le même "
                       "déséquilibre : le gain de +0,124 vient donc du contenu, pas du biais.")

            st.divider()

            # ===== 2. Illustration : re-ranking hybride réussite + pertinence =====
            st.subheader("Recommandation cold-start item — hybride réussite + pertinence")
            st.write("Pour un cours nouveau, le seul signal disponible est son **contenu**. "
                     "On combine deux signaux : la **réussite prédite** par EERNN-M et la "
                     "**pertinence thématique** (similarité de contenu avec l'historique). "
                     "Le curseur α arbitre entre les deux.")
            name2idx = dict(zip(courses["name"], courses["course_idx"]))
            all_names = list(name2idx.keys())

            hist_courses = st.multiselect("Historique de l'apprenant",
                                          all_names, key="eernn_hist")
            new_courses = st.multiselect(
                "Cours nouveaux à classer (catalogue)",
                [n for n in all_names if n not in hist_courses], key="eernn_new_set")
            alpha = st.slider("α — pondération (0 = pertinence pure · 1 = réussite pure)",
                              0.0, 1.0, 0.5, 0.1, key="eernn_alpha")

            if len(hist_courses) > 0 and len(new_courses) > 0:
                hist_idx = [int(name2idx[n]) for n in hist_courses]
                new_idx = [int(name2idx[n]) for n in new_courses]

                success = eernn_rank(model, meta, hist_idx, new_idx)
                relevance = thematic_relevance(S1, hist_idx, new_idx)

                s_n = _minmax(success)
                r_n = _minmax(relevance)
                hybrid = alpha * s_n + (1 - alpha) * r_n

                out = pd.DataFrame({
                    "Cours nouveau": new_courses,
                    "Réussite prédite": success.round(3),
                    "Pertinence (cosinus)": relevance.round(3),
                    "Score hybride": hybrid.round(3),
                })
                if stats is not None:
                    s = stats.set_index("course_idx")
                    out["Taux de base (contexte)"] = [
                        round(float(s.loc[i, "base_success_rate"]), 3)
                        if i in s.index else np.nan for i in new_idx]
                out = out.sort_values("Score hybride", ascending=False)
                out.insert(0, "Rang", range(1, len(out) + 1))
                st.dataframe(
                    out, use_container_width=True, hide_index=True,
                    column_config={
                        "Réussite prédite": st.column_config.ProgressColumn(
                            "Réussite prédite", format="%.3f",
                            min_value=0.0, max_value=1.0),
                        "Pertinence (cosinus)": st.column_config.NumberColumn(
                            "Pertinence (cosinus)", format="%.3f"),
                        "Score hybride": st.column_config.ProgressColumn(
                            "Score hybride", format="%.3f",
                            min_value=0.0, max_value=1.0),
                    })
                st.caption("Déplace α pour voir le classement basculer : vers 1, les cours "
                           "« faciles » (forte réussite) dominent ; vers 0, les cours "
                           "thématiquement proches de l'historique remontent. La réussite est "
                           "validée par AUC ; la pertinence est un signal de contenu "
                           "(sans vérité-terrain, donc non chiffré) — approche standard du "
                           "cold-start item où le contenu est le seul signal disponible.")


# ----------------------------------------------------------------------
# ONGLET 3 — Architecture & preuves
# ----------------------------------------------------------------------
with tab3:
    eyebrow("Passage à l'échelle · pipeline Big Data")
    st.header("Architecture & preuves")
    st.write("Le pipeline Big Data (Medallion Bronze→Silver→Gold, Spark/HDFS) est un "
             "traitement **batch** : il se présente par ses preuves, pas en direct.")
    figs = sorted((ART).glob("fig_*.png")) if ART.exists() else []
    archi = ART / "architecture.png"
    if archi.exists():
        st.image(str(archi), caption="Architecture générale de la plateforme")
    if figs:
        st.subheader("Figures des expériences")
        cols = st.columns(2)
        for i, f in enumerate(figs):
            cols[i % 2].image(str(f), caption=f.stem)

    st.subheader("Résultats clés")
    st.dataframe(
        pd.DataFrame({
            "Brique": ["GB-LTR (nominal)", "Item cold-start (EERNN-M)",
                       "Item baseline (LSTM-M)", "User cold-start (DKT)"],
            "Métrique": ["AUC 0,8818 · NDCG@10 0,9829", "AUC 0,9589 ± 0,0016",
                         "AUC 0,8353 ± 0,0088", "AUC 0,9662 ± 0,0064"],
        }),
        use_container_width=True, hide_index=True)