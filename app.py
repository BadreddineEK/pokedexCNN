import streamlit as st
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go

from pokemon_info import POKEMON_CLASSES, POKEMON_INFO, TYPE_COLORS

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PokéMAM — Pokédex CNN",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; text-align: center; }
    .subtitle   { text-align: center; color: #666; margin-bottom: 1.5rem; }
    .pokemon-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        border-radius: 16px; padding: 1.5rem;
        border: 2px solid #dde1e7;
        text-align: center;
    }
    .type-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    - Classes : **10 Pokémon**
    - Précision : **~88%**
    """)
    st.divider()
    st.markdown("### 🐾 Pokémon reconnus")
    for name in POKEMON_CLASSES:
        info = POKEMON_INFO[name]
        st.markdown(f"{info['emoji']} `{info['numero']}` {name}")

# ── En-tête principal ─────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎮 PokéMAM — Pokédex CNN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Reconnaissance de Pokémon par réseau de neurones convolutif · Polytech Lyon 2023</p>', unsafe_allow_html=True)

# ── Chargement du modèle ──────────────────────────────────────────────────────
# Le modèle est téléchargé depuis Google Drive au premier démarrage
# → Remplacez GDRIVE_FILE_ID par l'ID de votre fichier model.keras partagé
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "")
MODEL_PATH = "model.keras"

@st.cache_resource(show_spinner="⏳ Téléchargement du modèle...")
def load_model(gdrive_id: str):
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            return None, f"Erreur téléchargement : {e}"
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Erreur chargement : {e}"

model = None
if not GDRIVE_FILE_ID:
    st.warning("""
    ⚠️ **Modèle non configuré**

    Pour activer les prédictions, deux options :

    **Option A — Streamlit Cloud :** ajoutez votre secret dans `.streamlit/secrets.toml` :
    ```toml
    GDRIVE_FILE_ID = "votre_id_google_drive"
    ```
    Puis partagez `model.keras` en accès public sur Google Drive.

    **Option B — En local :** placez `model.keras` à la racine et relancez l'app.
    """)
elif os.path.exists(MODEL_PATH):
    # Fichier déjà présent localement (ex: run local)
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Modèle chargé !")
    except Exception as e:
        st.error(f"❌ {e}")
else:
    model, err = load_model(GDRIVE_FILE_ID)
    if err:
        st.error(f"❌ {err}")
    else:
        st.success("✅ Modèle chargé !")

st.divider()

# ── Upload et prédiction ──────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown("### 📤 Charger une image")
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
                img_array = np.array(img_resized) / 255.0
                img_input = np.expand_dims(img_array, axis=0)
                with st.spinner("Analyse en cours..."):
                    predictions = model.predict(img_input)[0]
                st.session_state["predictions"] = predictions
        else:
            st.info("ℹ️ Configurez le modèle (voir avertissement ci-dessus) pour activer les prédictions.")

with col_result:
    st.markdown("### 📊 Résultat")

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]
        top_idx = int(np.argmax(predictions))
        top_name = POKEMON_CLASSES[top_idx]
        top_conf = predictions[top_idx] * 100
        info = POKEMON_INFO[top_name]

        types_html = "".join(
            f'<span class="type-badge" style="background:{TYPE_COLORS.get(t, "#888")}">{t}</span>'
            for t in info["types"]
        )
        st.markdown(f"""
        <div class="pokemon-card">
            <h2>{info['emoji']} {top_name} {info['numero']}</h2>
            <div>{types_html}</div>
            <h3 style="color:#2196F3">Confiance : {top_conf:.1f}%</h3>
            <p style="color:#555; font-style:italic">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        with st.expander("📈 Statistiques de base", expanded=True):
            stats = {
                "HP": info["hp"], "ATK": info["atk"], "DEF": info["def"],
                "ATK Spé": info["spa"], "DEF Spé": info["spd"], "Vitesse": info["spe"]
            }
            for stat_name, val in stats.items():
                c1, c2 = st.columns([1, 3])
                c1.markdown(f"**{stat_name}**")
                c2.progress(min(val / 160, 1.0), text=str(val))

        st.markdown("### 🎯 Distribution des probabilités")
        probs_pct = [p * 100 for p in predictions]
        colors = [TYPE_COLORS.get(POKEMON_INFO[n]["types"][0], "#888") for n in POKEMON_CLASSES]

        fig = go.Figure(go.Bar(
            x=POKEMON_CLASSES,
            y=probs_pct,
            marker_color=colors,
            marker_line_color="white",
            marker_line_width=1.5,
            text=[f"{p:.1f}%" for p in probs_pct],
            textposition="outside",
        ))
        fig.update_layout(
            title="Probabilités par classe (softmax)",
            xaxis_title="Pokémon",
            yaxis_title="Probabilité (%)",
            yaxis_range=[0, 115],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            showlegend=False,
            height=400,
            margin=dict(t=50, b=10),
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Chargez une image et cliquez sur **Identifier le Pokémon** pour commencer.")
        st.markdown("### 🐾 Pokémon reconnus par le modèle")
        cols = st.columns(5)
        for i, name in enumerate(POKEMON_CLASSES):
            info = POKEMON_INFO[name]
            with cols[i % 5]:
                types_str = " · ".join(info["types"])
                st.markdown(f"""
                <div style="text-align:center; padding:8px; border-radius:10px;
                            background:#f8f9fa; margin:4px; border:1px solid #e0e0e0">
                    <div style="font-size:2rem">{info['emoji']}</div>
                    <strong>{name}</strong><br>
                    <small style="color:#666">{info['numero']}</small><br>
                    <small>{types_str}</small>
                </div>
                """, unsafe_allow_html=True)

st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem'>"
    "Made with ❤️ and a lot of Pokéballs · Polytech Lyon 2023</p>",
    unsafe_allow_html=True
)
