import streamlit as st
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go

from pokemon_info import POKEMON_CLASSES, POKEMON_INFO, TYPE_COLORS

st.set_page_config(
    page_title="PokéMAM — Pokédex CNN",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.6rem; font-weight: 800; text-align: center; }
    .subtitle   { text-align: center; color: #666; margin-bottom: 1rem; }
    .pokemon-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        border-radius: 16px; padding: 1.5rem;
        border: 2px solid #dde1e7; text-align: center;
    }
    .type-badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        color: white; font-weight: 600; font-size: 0.85rem; margin: 2px;
    }
    .cnn-box {
        border: 2px solid #dde1e7; border-radius: 12px; padding: 12px 16px;
        margin: 6px 0; text-align: center; font-size: 0.9rem;
    }
    .info-box {
        background: #f0f4ff; border-left: 4px solid #4a90d9;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 PokéMAM")
    st.markdown("**Pokédex par CNN**")
    st.divider()
    st.markdown("### 📚 Projet académique")
    st.markdown("""
    - 🏫 **Polytech Lyon** — MAM 4A
    - 📅 **2022 / 2023**
    - 👥 **Équipe :**
      - EL KHALFIOUI Nadir
      - EL KHAMLICHI Badreddine
      - LEHLALI Hédi
    """)
    st.divider()
    st.markdown("### 🤖 Modèle")
    st.markdown("""
    - Architecture : **CNN custom**
    - Framework : **TensorFlow / Keras**
    - Images : **200×200 px**
    - Entraînement : **10 epochs**
    - Classes : **10 Pokémon**
    """)
    st.divider()
    st.markdown("### ⚠️ Pokémon reconnus")
    st.markdown("> Le modèle ne reconnaît **que ces 10 Pokémon**. Toute autre image donnera un résultat incorrect.")
    for name in POKEMON_CLASSES:
        info = POKEMON_INFO[name]
        st.markdown(f"{info['emoji']} `{info['numero']}` **{name}**")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎮 PokéMAM — Pokédex CNN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Reconnaissance de Pokémon par réseau de neurones convolutif · Polytech Lyon 2023</p>', unsafe_allow_html=True)

# ── Chargement du modèle ───────────────────────────────────────────────────
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

# ── Onglets ────────────────────────────────────────────────────────────────
tab_pred, tab_cnn, tab_about = st.tabs(["🔍 Prédiction", "🧠 Architecture CNN", "📊 À propos du dataset"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 : PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### 📤 Identifier un Pokémon")

    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Important :</strong> ce modèle a été entraîné <strong>uniquement sur 10 Pokémon</strong>
    (voir la barre latérale). Uploadez une image de l'un de ces 10 Pokémon pour obtenir un résultat fiable.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 2], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Déposez l'image d'un Pokémon",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formats acceptés : JPG, PNG, WEBP",
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Image chargée", use_container_width=True)

            if model is not None:
                if st.button("🔍 Identifier le Pokémon", type="primary", use_container_width=True):
                    img_resized = img_pil.resize((200, 200))
                    img_array  = np.array(img_resized).astype("float32")
                    img_input  = np.expand_dims(img_array, axis=0)
                    with st.spinner("Analyse en cours..."):
                        predictions = model.predict(img_input)[0]
                    st.session_state["predictions"] = predictions
            else:
                st.info("ℹ️ Modèle non chargé — prédictions désactivées.")

        # Galerie des 10 Pokémon supportés
        st.divider()
        st.markdown("**Pokémon supportés :**")
        cols5 = st.columns(5)
        for i, name in enumerate(POKEMON_CLASSES):
            info = POKEMON_INFO[name]
            with cols5[i % 5]:
                st.markdown(
                    f'<div style="text-align:center;padding:6px;border-radius:8px;'
                    f'background:#f8f9fa;margin:3px;border:1px solid #e0e0e0">'
                    f'<div style="font-size:1.6rem">{info["emoji"]}</div>'
                    f'<small><b>{name}</b></small><br>'
                    f'<small style="color:#888">{info["numero"]}</small></div>',
                    unsafe_allow_html=True,
                )

    with col_result:
        st.markdown("### 📊 Résultat")

        if "predictions" in st.session_state:
            predictions = st.session_state["predictions"]
            top_idx  = int(np.argmax(predictions))
            top_name = POKEMON_CLASSES[top_idx]
            top_conf = predictions[top_idx] * 100
            info     = POKEMON_INFO[top_name]

            types_html = "".join(
                f'<span class="type-badge" style="background:{TYPE_COLORS.get(t, "#888")}">{t}</span>'
                for t in info["types"]
            )
            legendary_badge = "⭐ Légendaire" if info["legendaire"] else ""
            st.markdown(f"""
            <div class="pokemon-card">
                <h2>{info["emoji"]} {top_name} {info["numero"]}</h2>
                <div>{types_html}</div>
                <h3 style="color:#2196F3">Confiance : {top_conf:.1f}%</h3>
                <p style="color:#555; font-style:italic">{info["description"]}</p>
                <p>{legendary_badge}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📈 Statistiques de base", expanded=True):
                stats = {
                    "HP": info["hp"], "ATK": info["atk"], "DEF": info["def"],
                    "Sp.ATK": info["spa"], "Sp.DEF": info["spd"], "Speed": info["spe"],
                }
                for stat_name, val in stats.items():
                    c1, c2 = st.columns([1, 3])
                    c1.markdown(f"**{stat_name}**")
                    c2.progress(min(val / 160, 1.0), text=str(val))

            st.markdown("### 🎯 Distribution des probabilités")
            probs_pct = [p * 100 for p in predictions]
            colors    = [TYPE_COLORS.get(POKEMON_INFO[n]["types"][0], "#888") for n in POKEMON_CLASSES]
            fig = go.Figure(go.Bar(
                x=POKEMON_CLASSES, y=probs_pct,
                marker_color=colors, marker_line_color="white", marker_line_width=1.5,
                text=[f"{p:.1f}%" for p in probs_pct], textposition="outside",
            ))
            fig.update_layout(
                title="Probabilités par classe (softmax)",
                xaxis_title="Pokémon", yaxis_title="Probabilité (%)",
                yaxis_range=[0, 115],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12), showlegend=False, height=380,
                margin=dict(t=50, b=10),
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Chargez une image et cliquez sur **Identifier le Pokémon** pour commencer.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 : ARCHITECTURE CNN
# ═══════════════════════════════════════════════════════════════════════════
with tab_cnn:
    st.markdown("## 🧠 Architecture du réseau de neurones convolutif (CNN)")

    st.markdown("""
    Un **CNN (Convolutional Neural Network)** est un réseau de neurones spécialement conçu pour
    traiter des images. Il apprend automatiquement à reconnaître des motifs visuels (bords, formes,
    textures) à partir des données d'entraînement, sans qu'on ait besoin de définir manuellement
    les caractéristiques.
    """)

    # Pipeline visuel
    st.markdown("### 🔄 Pipeline complet")
    steps = [
        ("📷", "Image brute (variable)",        "Sprites PNG des Pokémon"),
        ("🔧", "Prétraitement",                  "Redimensionnement 200×200 px · Data augmentation\n(rotation, zoom, flip) pour générer ×30 images/Pokémon"),
        ("📂", "Split train/val/test",            "70% entraînement · 20% test · 10% validation\n(généré depuis le sprite avec augmentation)"),
        ("🧱", "Conv2D 128 filtres (4×4) + ReLU", "Détection de motifs locaux bas niveau (bords, couleurs)"),
        ("⬇️", "MaxPooling2D",                    "Réduction spatiale × 2 → compression de l'info"),
        ("🧱", "Conv2D 64 filtres (4×4) + ReLU",  "Motifs de niveau intermédiaire (formes, textures)"),
        ("⬇️", "MaxPooling2D",                    "Réduction spatiale × 2"),
        ("🧱", "Conv2D 32 filtres (4×4) + ReLU",  "Motifs complexes (parties du corps, silhouettes)"),
        ("⬇️", "MaxPooling2D",                    "Réduction spatiale × 2"),
        ("🧱", "Conv2D 16 filtres (4×4) + ReLU",  "Caractéristiques de haut niveau"),
        ("⬇️", "MaxPooling2D",                    "Réduction spatiale × 2"),
        ("📐", "Flatten",                         "Vecteur 1D à partir de la carte 3D"),
        ("🔗", "Dense 64 + ReLU",                 "Couche entièrement connectée — combine les features"),
        ("🎯", "Dense 10 + Softmax",              "Sortie : 10 probabilités (une par Pokémon)"),
    ]
    for icon, name, desc in steps:
        col_i, col_c = st.columns([1, 5])
        col_i.markdown(f"<div style='font-size:2rem;text-align:center'>{icon}</div>", unsafe_allow_html=True)
        col_c.markdown(f"<div class='cnn-box'><strong>{name}</strong><br><small style='color:#555'>{desc}</small></div>", unsafe_allow_html=True)
        if icon in ("🔧", "📂"):
            st.markdown("")
        elif "MaxPooling" not in name and "Dense 10" not in name:
            st.markdown("<div style='text-align:center;font-size:1.2rem;color:#aaa'>↓</div>", unsafe_allow_html=True)

    st.divider()

    # Hyperparamètres
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### ⚙️ Hyperparamètres")
        st.markdown("""
        | Paramètre | Valeur |
        |-----------|--------|
        | Image size | 200 × 200 px |
        | Batch size | 32 |
        | Epochs | 10 |
        | Optimiseur | Adam |
        | Loss | SparseCategoricalCrossentropy |
        | Activation finale | Softmax |
        """)
    with col_b:
        st.markdown("### 📊 Data augmentation")
        st.markdown("""
        Comme chaque Pokémon n'a **qu'un seul sprite** de base, on génère **30 variantes** d'entraînement
        par Pokémon par augmentation :

        - Rotation ±20°
        - Zoom ±15%
        - Décalage H/V ±10%
        - Flip horizontal
        - Variation de luminosité ±15%
        """)
    with col_c:
        st.markdown("### 🎯 Résultat")
        st.markdown("""
        Le modèle entraîné est sauvegardé en `.keras` (TensorFlow 2.x).

        - **Classes** : 10 Pokémon
        - **Précision** : ~88%
        - **Taille du modèle** : ~3 MB

        La prédiction finale est l'index du **score softmax le plus élevé**,
        converti en nom de classe via `POKEMON_CLASSES`.
        """)

    st.divider()
    st.markdown("### 🔬 Pourquoi ces choix d'architecture ?")
    st.markdown("""
    - **4 blocs Conv+Pool** : suffisant pour des sprites 200×200 relativement simples (fond blanc, formes distinctes).
      Plus de couches augmenteraient le temps d'entraînement sans gain notable sur si peu de données.
    - **Filtres décroissants (128→64→32→16)** : les premières couches détectent beaucoup de motifs bas-niveau,
      les suivantes affinent en réduisant la dimensionnalité.
    - **Softmax + SparseCategoricalCrossentropy** : idéal pour la classification multi-classes avec labels entiers.
    - **Adam** : optimiseur adaptatif, converge rapidement même avec peu de données.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 : À PROPOS DU DATASET
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## 📊 À propos du dataset et du projet")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🗂️ Dataset utilisé (version actuelle)")
        st.markdown("""
        **Source :** sprites PNG officieux, un fichier par Pokémon.

        - **Format** : 1 sprite PNG par Pokémon (fond transparent/blanc)
        - **~900 Pokémon** disponibles dans `/images/`
        - **10 sélectionnés** pour l'entraînement (voir ci-dessous)
        - Images redimensionnées à **200×200 px** avant entraînement
        - Data augmentation × 30 par Pokémon → **300 images d'entraînement**

        **Pourquoi 10 seulement ?**
        Avec un seul sprite de base par Pokémon, même avec augmentation, entraîner
        sur 900 classes (~27 000 images) prendrait plusieurs heures et nécessiterait
        un GPU. 10 classes permettent de démontrer le pipeline complet rapidement.
        """)

        st.markdown("### ⚠️ Les 10 Pokémon retenus")
        st.markdown("> Choisis pour leur **diversité visuelle** et leur popularité.")
        for i, name in enumerate(POKEMON_CLASSES):
            info = POKEMON_INFO[name]
            types_str = " · ".join(
                f'<span style="background:{TYPE_COLORS.get(t,"#888")};color:white;'
                f'padding:1px 7px;border-radius:10px;font-size:0.8rem">{t}</span>'
                for t in info["types"]
            )
            st.markdown(
                f"**{i}.** {info['emoji']} **{name}** `{info['numero']}`  &nbsp; {types_str}",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 🚀 Pour aller plus loin — Dataset Kaggle")
        st.markdown("""
        Pour entraîner sur **plus de Pokémon** ou avec **plus d'images** par classe,
        utilisez le dataset officiel Kaggle :

        > **[Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)**
        > par *vishalsubbiah* sur Kaggle

        Ce dataset contient :
        - **809 Pokémon** (générations 1 à 7)
        - **1 image PNG** par Pokémon (sprites haute qualité)
        - **Un CSV** avec le type de chaque Pokémon
        - Licence : CC0 (domaine public)
        """)

        st.link_button(
            "📥 Télécharger sur Kaggle",
            "https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types",
        )

        st.markdown("### 📈 Comment étendre le modèle ?")
        st.markdown("""
        Dans le notebook `pokemam_10_epoch.ipynb`, modifiez simplement :

        ```python
        # Cellule 10 — changer les 10 Pokémon
        SELECTED_POKEMON = [
            'pikachu', 'charizard', 'mewtwo', 'gengar', 'eevee',
            'lucario', 'gardevoir', 'gyarados', 'snorlax', 'blastoise',
            # Ajouter autant que voulu...
        ]
        ```

        Et ajustez les hyperparamètres :

        ```python
        N_TRAIN = 50    # plus d'images augmentées
        epochs  = 20   # plus d'epochs
        ```

        Puis re-sauvegardez le modèle :
        ```python
        model.save('pokemon_cnn_model.keras')
        ```
        """)

        st.divider()
        st.markdown("### 🗂️ Structure du dépôt Git")
        st.markdown("""
        ```
        pokedexCNN/
        ├── app.py                     # App Streamlit (ce fichier)
        ├── pokemon_info.py            # Métadonnées des 10 Pokémon
        ├── pokemon_cnn_model.keras    # Modèle entraîné (~3 MB)
        ├── pokemon.csv                # Données Pokémon (types, évolutions)
        ├── pokemam_10_epoch.ipynb     # Notebook d'entraînement
        ├── requirements.txt
        ├── .gitignore
        │
        ├── images/        🚫 gitignore — sprites bruts (~900 PNG)
        ├── train/         🚫 gitignore — images augmentées
        ├── val/           🚫 gitignore — images augmentées
        ├── test/          🚫 gitignore — images augmentées
        └── dataset/       🚫 gitignore — données brutes
        ```
        > Les dossiers `images/`, `train/`, `val/`, `test/` sont exclus du git.
        > Ils se régénèrent en exécutant le notebook.
        """)

st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem'>"
    "Made with ❤️ and a lot of Pokéballs · Polytech Lyon 2023 · "
    "<a href='https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types' "
    "style='color:#aaa'>Dataset Kaggle</a></p>",
    unsafe_allow_html=True,
)
