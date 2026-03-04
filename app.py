import streamlit as st
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go

from pokemon_info import POKEMON_CLASSES, POKEMON_INFO, TYPE_COLORS

st.set_page_config(
    page_title="PokéDex CNN",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header ─────────────────────────────────────────────────────── */
    .hero-title {
        font-size: 3rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #E63946 0%, #FF6B35 50%, #F72585 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center; color: #888; font-size: 1rem;
        margin-bottom: 0.2rem; letter-spacing: 0.03em;
    }
    .hero-badge {
        display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .badge {
        background: #f0f2f6; color: #444; border-radius: 20px;
        padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
        border: 1px solid #dde1e7;
    }

    /* ── Cards ──────────────────────────────────────────────────────── */
    .pokemon-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px; padding: 1.8rem 1.5rem;
        border: 1px solid #30304a; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .pokemon-card h2 { color: #fff; margin-bottom: 0.5rem; }
    .pokemon-card h3 { margin-top: 0.8rem; }
    .pokemon-card p  { color: #aaa; font-style: italic; margin-top: 0.6rem; }

    .type-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        color: white; font-weight: 700; font-size: 0.82rem;
        margin: 3px; letter-spacing: 0.04em; text-transform: uppercase;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }

    /* ── Pokémon miniature ──────────────────────────────────────────── */
    .poke-mini {
        text-align: center; padding: 8px 4px; border-radius: 12px;
        background: #f8f9fa; margin: 3px;
        border: 1.5px solid #e8ecf0;
        transition: border-color 0.2s;
    }
    .poke-mini:hover { border-color: #E63946; }

    /* ── Info/Warning boxes ─────────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #e8f4ff 0%, #f0f8ff 100%);
        border-left: 4px solid #2196F3; padding: 12px 16px;
        border-radius: 0 10px 10px 0; margin: 10px 0; font-size: 0.9rem;
    }
    .warn-box {
        background: linear-gradient(135deg, #fff8e1 0%, #fffde7 100%);
        border-left: 4px solid #FFC107; padding: 12px 16px;
        border-radius: 0 10px 10px 0; margin: 10px 0; font-size: 0.9rem;
    }

    /* ── CNN pipeline ───────────────────────────────────────────────── */
    .cnn-layer {
        border-radius: 12px; padding: 10px 16px; margin: 4px 0;
        font-size: 0.88rem; border: 1.5px solid;
    }
    .cnn-conv   { background: #fff0f0; border-color: #E63946; }
    .cnn-pool   { background: #f0fff4; border-color: #2ecc71; }\n    .cnn-dense  { background: #f0f4ff; border-color: #4a90d9; }
    .cnn-out    { background: linear-gradient(135deg, #fff0f8, #f0f4ff); border-color: #F72585; }
    .cnn-pre    { background: #fafafa; border-color: #aaa; }
    .cnn-arrow  { text-align: center; color: #ccc; font-size: 1.3rem; line-height: 1.2; }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    .sidebar-section {
        background: #f8f9fa; border-radius: 12px;
        padding: 12px 14px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎮 PokéDex CNN")
    st.caption("Modernisation d'un projet école en app interactive")
    st.divider()

    st.markdown("**🤖 Modèle**")
    st.markdown("""
    - Architecture : **CNN custom**
    - Framework : **TensorFlow / Keras**
    - Images : **200×200 px**
    - Epochs : **10**
    - Classes : **10 Pokémon**
    - Précision : **~88%**
    """)
    st.divider()

    st.markdown("**⚠️ Pokémon reconnus**")
    st.caption("Le modèle ne reconnaît **que ces 10 Pokémon**.")
    for name in POKEMON_CLASSES:
        info = POKEMON_INFO[name]
        st.markdown(f"{info['emoji']} `{info['numero']}` {name}")

    st.divider()
    st.markdown("**📎 Liens**")
    st.markdown("[📊 Dataset Kaggle](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)")
    st.markdown("[📓 Notebook GitHub](https://github.com/BadreddineEK/pokedexCNN/blob/main/pokemam_10_epoch.ipynb)")

# ══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🎮 PokéDex CNN</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Reconnaissance de Pokémon par réseau de neurones convolutif</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="hero-badge">
  <span class="badge">🏫 Polytech Lyon • 2022‣2023</span>
  <span class="badge">🔁 Repris &amp; modernizé</span>
  <span class="badge">🤖 TensorFlow / Keras</span>
  <span class="badge">🎯 10 classes • ~88%</span>
  <span class="badge">✨ Streamlit Cloud</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH = "pokemon_cnn_model.keras"

@st.cache_resource(show_spinner="⏳ Chargement du modèle CNN...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
else:
    st.warning(f"⚠️ Modèle introuvable (`{MODEL_PATH}`). Entraînez et sauvegardez le modèle depuis le notebook.")

# ══════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tab_pred, tab_cnn, tab_about = st.tabs([
    "🔍 Identifier un Pokémon",
    "🧠 Architecture CNN",
    "📊 Dataset & Projet",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 : PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Scope du modèle :</strong> entraîné sur <strong>10 Pokémon seulement</strong>.
    Uploadez l’image de l’un d’eux pour un résultat fiable (voir liste en sidebar).
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 2], gap="large")

    with col_upload:
        st.markdown("#### 📤 Uploader une image")
        uploaded = st.file_uploader(
            "Formats acceptés : JPG, PNG, WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, use_container_width=True)
            if model is not None:
                if st.button("🔍 Identifier", type="primary", use_container_width=True):
                    arr = np.expand_dims(
                        np.array(img_pil.resize((200, 200))).astype("float32"), 0
                    )
                    with st.spinner("Analyse en cours..."):
                        preds = model.predict(arr)[0]
                    st.session_state["predictions"] = preds
            else:
                st.info("ℹ️ Modèle non chargé.")

        st.divider()
        st.markdown("**Pokémon supportés**")
        cols5 = st.columns(5)
        for i, name in enumerate(POKEMON_CLASSES):
            info = POKEMON_INFO[name]
            with cols5[i % 5]:
                st.markdown(
                    f'<div class="poke-mini">'
                    f'<div style="font-size:1.5rem">{info["emoji"]}</div>'
                    f'<div style="font-size:0.72rem;font-weight:600">{name}</div>'
                    f'<div style="font-size:0.65rem;color:#999">{info["numero"]}</div></div>',
                    unsafe_allow_html=True,
                )

    with col_result:
        st.markdown("#### 📊 Résultat de la prédiction")

        if "predictions" in st.session_state:
            preds    = st.session_state["predictions"]
            top_idx  = int(np.argmax(preds))
            top_name = POKEMON_CLASSES[top_idx]
            top_conf = preds[top_idx] * 100
            info     = POKEMON_INFO[top_name]

            confidence_color = (
                "#2ecc71" if top_conf >= 80 else
                "#FFC107" if top_conf >= 50 else "#E63946"
            )

            types_html = "".join(
                f'<span class="type-badge" style="background:{TYPE_COLORS.get(t, "#888")}">{t}</span>'
                for t in info["types"]
            )
            legendary = "⭐ Légendaire" if info["legendaire"] else ""

            st.markdown(f"""
            <div class="pokemon-card">
                <div style="font-size:3.5rem">{info['emoji']}</div>
                <h2>{top_name} <span style="color:#888;font-size:1rem">{info['numero']}</span></h2>
                <div>{types_html}</div>
                <h3 style="color:{confidence_color};margin-top:1rem">
                    Confiance : {top_conf:.1f}%
                </h3>
                <p>{info['description']}</p>
                <p style="color:#f1c40f;font-size:0.9rem">{legendary}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            with st.expander("📊 Stats de base", expanded=True):
                stats = {
                    "HP": info["hp"], "ATK": info["atk"], "DEF": info["def"],
                    "Sp.ATK": info["spa"], "Sp.DEF": info["spd"], "Speed": info["spe"],
                }
                total = sum(stats.values())
                st.caption(f"Total : **{total}**")
                for stat_name, val in stats.items():
                    c1, c2 = st.columns([1, 3])
                    c1.markdown(f"**{stat_name}**")
                    c2.progress(min(val / 160, 1.0), text=str(val))

            st.markdown("#### 🎯 Distribution softmax")
            probs_pct = [p * 100 for p in preds]
            colors    = [TYPE_COLORS.get(POKEMON_INFO[n]["types"][0], "#888") for n in POKEMON_CLASSES]
            fig = go.Figure(go.Bar(
                x=POKEMON_CLASSES, y=probs_pct, marker_color=colors,
                marker_line_color="rgba(255,255,255,0.3)", marker_line_width=1,
                text=[f"{p:.1f}%" for p in probs_pct], textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="Pokémon", yaxis_title="Probabilité (%)",
                yaxis_range=[0, 118],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11),
                showlegend=False, height=360,
                margin=dict(t=10, b=10, l=0, r=0),
            )
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#aaa">
                <div style="font-size:4rem">👀</div>
                <p>Uploadez une image et cliquez sur <strong>Identifier</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 : ARCHITECTURE CNN
# ─────────────────────────────────────────────────────────────────────────
with tab_cnn:
    st.markdown("## 🧠 Architecture du réseau de neurones convolutif")
    st.markdown("""
    Un **CNN (Convolutional Neural Network)** apprend automatiquement à extraire des caractéristiques
    visuelles hiérarchiques depuis une image — bords et couleurs dans les premières couches,
    formes complexes et silhouettes dans les dernières — sans qu’on ait besoin de les définir manuellement.
    """)

    col_pipeline, col_params = st.columns([3, 2], gap="large")

    with col_pipeline:
        st.markdown("### 🔄 Pipeline")

        layers = [
            ("cnn-pre",  "📷 Image brute",                    "Sprite PNG du Pokémon (taille variable)"),
            ("cnn-pre",  "🔧 Prétraitement + Data augmentation", "Redim. 200×200 · Rotation ±20° · Zoom ±15% · Flip · Luminosité ·15%"),
            ("cnn-pre",  "📂 Split 70/20/10",                  "Train 70% · Test 20% · Validation 10%"),
            None,  # séparateur visuel
            ("cnn-conv", "🧱 Conv2D • 128 filtres 4×4 • ReLU",  "Détection motifs bas niveau : bords, couleurs, gradients"),
            ("cnn-pool", "↓ MaxPooling2D",                       "Réduction spatiale ×2 — conserve les réponses max"),
            ("cnn-conv", "🧱 Conv2D • 64 filtres 4×4 • ReLU",   "Motifs intermédiaires : formes, textures"),
            ("cnn-pool", "↓ MaxPooling2D",                       "Réduction spatiale ×2"),
            ("cnn-conv", "🧱 Conv2D • 32 filtres 4×4 • ReLU",   "Motifs complexes : parties du corps, silhouettes"),
            ("cnn-pool", "↓ MaxPooling2D",                       "Réduction spatiale ×2"),
            ("cnn-conv", "🧱 Conv2D • 16 filtres 4×4 • ReLU",   "Caractéristiques de haut niveau"),
            ("cnn-pool", "↓ MaxPooling2D",                       "Réduction spatiale ×2"),
            None,
            ("cnn-dense", "📏 Flatten",                          "Carte 3D → vecteur 1D"),
            ("cnn-dense", "🔗 Dense 64 • ReLU",                  "Couche fully-connected — combine toutes les features"),
            ("cnn-out",   "🎯 Dense 10 • Softmax",               "10 probabilités de sortie — une par Pokémon"),
        ]

        for layer in layers:
            if layer is None:
                st.markdown("<div class='cnn-arrow'>⋮</div>", unsafe_allow_html=True)
            else:
                cls, name, desc = layer
                st.markdown(
                    f"<div class='cnn-layer {cls}'>"
                    f"<strong>{name}</strong>"
                    f"<br><span style='color:#666;font-size:0.82rem'>{desc}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    with col_params:
        st.markdown("### ⚙️ Hyperparamètres")
        st.markdown("""
        | Paramètre | Valeur |
        |---|---|
        | Taille image | 200×200 px |
        | Batch size | 32 |
        | Epochs | 10 |
        | Optimiseur | **Adam** |
        | Loss | SparseCategorical CE |
        | Activation sortie | **Softmax** |
        """)

        st.markdown("### 📈 Entraînement")
        st.markdown("""
        Avec un seul sprite par Pokémon, la **data augmentation** est cruciale :
        chaque image est transformée pour générer **30 variantes** d'entraînement.

        Résultats obtenus en **~10 epochs** :
        - Précision validation : **~88%**
        - Taille modèle : **~3 MB**
        """)

        st.markdown("### 🔬 Pourquoi ces choix ?")
        st.markdown("""
        - **Filtres décroissants (128→64→32→16)** : les premières couches détectent
          beaucoup de motifs bas-niveau, les suivantes affinent.
        - **4 blocs Conv+Pool** suffisent pour des sprites simples
          (fond blanc, formes distinctives).
        - **Adam** converge vite même avec peu de données.
        - **Softmax** produit des probabilités interprétables pour 10 classes.
        """)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 : DATASET & PROJET
# ─────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("## 📊 Dataset & Projet")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📖 Contexte")
        st.markdown("""
        Ce projet a été réalisé en **4ème année à Polytech Lyon** (2022–2023)
        dans le cadre d’un cours de machine learning appliqué.

        L’objectif original était de démontrer le pipeline complet d’un CNN —
        prétraitement, augmentation, entraînement, évaluation — sur un jeu de
        données visuel concret.

        Cette version est une **reprise et modernisation** du projet original :
        le modèle est inchangé, mais l’interface a été entièrement reconstruite
        en Streamlit avec une expérience utilisateur améliorée et déployée
        sur **Streamlit Cloud**.
        """)

        st.markdown("### 🗂️ Dataset")
        st.markdown("""
        **Source :** [Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)
        sur Kaggle — licence CC0.

        - **809 Pokémon** (générations 1 à 7)
        - **1 sprite PNG** par Pokémon (fond transparent)
        - **CSV** avec types et métadonnées

        Pour ce projet, **10 Pokémon** ont été sélectionnés pour leur
        diversité visuelle (couleurs, silhouettes, types distincts).
        Chaque sprite génère **30 images augmentées** → 300 images d’entraînement.
        """)
        st.link_button("📥 Dataset Kaggle", "https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types")

        st.markdown("### ⚙️ 10 Pokémon sélectionnés")
        for i, name in enumerate(POKEMON_CLASSES):
            info = POKEMON_INFO[name]
            types_html = "".join(
                f'<span style="background:{TYPE_COLORS.get(t,"#888")};color:white;'
                f'padding:1px 8px;border-radius:10px;font-size:0.78rem;margin:2px">'
                f'{t}</span>' for t in info["types"]
            )
            st.markdown(
                f"**{i+1}.** {info['emoji']} **{name}** `{info['numero']}` &nbsp; {types_html}",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 🚀 Étendre le modèle")
        st.markdown("""
        Pour entraîner sur d’autres Pokémon ou plus de classes, ouvrez
        `pokemam_10_epoch.ipynb` et modifiez :

        ```python
        # Changer la liste des Pokémon
        SELECTED = [
            'pikachu', 'charizard', 'mewtwo',
            'gengar', 'eevee', 'lucario', ...
        ]
        # Augmenter les données
        N_AUGMENTED = 50    # images par Pokémon
        EPOCHS = 20
        ```
        Puis sauvegarder :
        ```python
        model.save('pokemon_cnn_model.keras')
        ```
        """)

        st.markdown("### 🗂️ Structure du repo")
        st.code("""
pokedexCNN/
├── app.py                    # App Streamlit
├── pokemon_info.py           # Méta-données des 10 Pokémon
├── pokemon_cnn_model.keras   # Modèle entraîné (~3 MB)
├── pokemon.csv               # Stats complètes (809 Pokémon)
├── pokemam_10_epoch.ipynb    # Notebook d'entraînement
├── requirements.txt
├── .streamlit/config.toml
│
├── images/   🚫 gitignore
├── train/    🚫 gitignore
├── val/      🚫 gitignore
└── test/     🚫 gitignore
        """, language="")

        st.markdown("### 🔗 Liens utiles")
        st.link_button("📓 Voir le notebook", "https://github.com/BadreddineEK/pokedexCNN/blob/main/pokemam_10_epoch.ipynb")
        st.link_button("⭐ GitHub du projet", "https://github.com/BadreddineEK/pokedexCNN")

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;color:#bbb;font-size:0.82rem'>"
    "Projet école Polytech Lyon 2022–2023 — repris &amp; modernizé · "
    "<a href='https://github.com/BadreddineEK/pokedexCNN' style='color:#bbb'>"
    "GitHub</a> · "
    "<a href='https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types' style='color:#bbb'>"
    "Dataset Kaggle</a></p>",
    unsafe_allow_html=True,
)
