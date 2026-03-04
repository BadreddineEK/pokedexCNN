import streamlit as st
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px

from pokemon_info import POKEMON_CLASSES, POKEMON_INFO, TYPE_COLORS

st.set_page_config(
    page_title="PokéDex CNN",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 3rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #E63946 0%, #FF6B35 50%, #F72585 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem;
    }
    .hero-sub   { text-align:center; color:#888; font-size:1rem; margin-bottom:0.2rem; letter-spacing:0.03em; }
    .hero-badge { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-bottom:1.5rem; }
    .badge      { background:#f0f2f6; color:#444; border-radius:20px; padding:3px 12px;
                  font-size:0.78rem; font-weight:600; border:1px solid #dde1e7; }

    /* -- Result card -- */
    .pokemon-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius:20px; padding:1.8rem 1.5rem;
        border:1px solid #30304a; text-align:center;
        box-shadow:0 8px 32px rgba(0,0,0,0.3);
    }
    .pokemon-card h2 { color:#fff; margin-bottom:0.5rem; }
    .pokemon-card p  { color:#aaa; font-style:italic; margin-top:0.6rem; }
    .type-badge {
        display:inline-block; padding:4px 14px; border-radius:20px;
        color:white; font-weight:700; font-size:0.82rem;
        margin:3px; letter-spacing:0.04em; text-transform:uppercase;
        box-shadow:0 2px 6px rgba(0,0,0,0.2);
    }
    .poke-mini { text-align:center; padding:8px 4px; border-radius:12px;
                 background:#f8f9fa; margin:3px; border:1.5px solid #e8ecf0; }

    /* -- Info / Warn boxes -- */
    .info-box { background:linear-gradient(135deg,#e8f4ff,#f0f8ff);
                border-left:4px solid #2196F3; padding:12px 16px;
                border-radius:0 10px 10px 0; margin:10px 0; font-size:0.9rem; }
    .warn-box { background:linear-gradient(135deg,#fff8e1,#fffde7);
                border-left:4px solid #FFC107; padding:12px 16px;
                border-radius:0 10px 10px 0; margin:10px 0; font-size:0.9rem; }

    /* -- CNN step cards -- */
    .step-card {
        border-radius:14px; padding:16px 20px; margin:6px 0;
        border:2px solid; transition: box-shadow 0.2s;
    }
    .step-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
    .step-input  { background:#fafafa;       border-color:#ccc;    }
    .step-aug    { background:#fff8f0;       border-color:#FF9800; }
    .step-conv   { background:#fff0f0;       border-color:#E63946; }
    .step-pool   { background:#f0fff4;       border-color:#2ecc71; }
    .step-dense  { background:#f0f4ff;       border-color:#4a90d9; }
    .step-out    { background:linear-gradient(135deg,#fff0f8,#f0f4ff); border-color:#F72585; }
    .step-title  { font-weight:700; font-size:1rem; margin-bottom:4px; }
    .step-sub    { color:#555; font-size:0.85rem; line-height:1.5; }
    .step-arrow  { text-align:center; color:#ccc; font-size:1.4rem; line-height:1; margin:2px 0; }

    /* -- Concept explainer -- */
    .concept-box {
        background:linear-gradient(135deg,#f8f9ff,#fff);
        border-radius:16px; padding:18px 20px;
        border:1px solid #e0e4f0; margin:8px 0;
    }
    .concept-title { font-size:1.05rem; font-weight:700; margin-bottom:6px; }
    .metaphor {
        background:#fffbea; border-left:3px solid #f1c40f;
        padding:8px 12px; border-radius:0 8px 8px 0;
        font-size:0.87rem; color:#555; margin-top:8px;
    }
</style>
""", unsafe_allow_html=True)

# ═══ SIDEBAR ════════════════════════════════════════════════════════════
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

# ═══ HERO ════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🎮 PokéDex CNN</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Reconnaissance de Pokémon par réseau de neurones convolutif</p>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-badge">
  <span class="badge">🏫 Polytech Lyon • 2022–2023</span>
  <span class="badge">🔁 Repris &amp; modernizé</span>
  <span class="badge">🤖 TensorFlow / Keras</span>
  <span class="badge">🎯 10 classes • ~88%</span>
  <span class="badge">✨ Streamlit Cloud</span>
</div>
""", unsafe_allow_html=True)

# ═══ CHARGEMENT MODÈLE ═══════════════════════════════════════════════════════
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
    st.warning(f"⚠️ Modèle introuvable (`{MODEL_PATH}`).")

# ═══ ONGLETS ════════════════════════════════════════════════════════════
tab_pred, tab_cnn, tab_about = st.tabs([
    "🔍 Identifier un Pokémon",
    "🧠 Comment ça marche ?",
    "📊 Dataset & Projet",
])

# ────────────────────────────────────────────────────────────────────────
# TAB 1 : PRÉDICTION
# ────────────────────────────────────────────────────────────────────────
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
            conf_color = "#2ecc71" if top_conf >= 80 else ("#FFC107" if top_conf >= 50 else "#E63946")
            types_html = "".join(
                f'<span class="type-badge" style="background:{TYPE_COLORS.get(t,"#888")}">{t}</span>'
                for t in info["types"]
            )
            st.markdown(f"""
            <div class="pokemon-card">
                <div style="font-size:3.5rem">{info['emoji']}</div>
                <h2>{top_name} <span style="color:#888;font-size:1rem">{info['numero']}</span></h2>
                <div>{types_html}</div>
                <h3 style="color:{conf_color};margin-top:1rem">Confiance : {top_conf:.1f}%</h3>
                <p>{info['description']}</p>
                <p style="color:#f1c40f;font-size:0.9rem">{'\u2b50 L\u00e9gendaire' if info['legendaire'] else ''}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            with st.expander("📊 Stats de base", expanded=True):
                stats = {"HP":info["hp"],"ATK":info["atk"],"DEF":info["def"],
                         "Sp.ATK":info["spa"],"Sp.DEF":info["spd"],"Speed":info["spe"]}
                st.caption(f"Total : **{sum(stats.values())}**")
                for sn, sv in stats.items():
                    c1, c2 = st.columns([1,3])
                    c1.markdown(f"**{sn}**")
                    c2.progress(min(sv/160,1.0), text=str(sv))
            st.markdown("#### 🎯 Distribution softmax")
            probs_pct = [p*100 for p in preds]
            colors    = [TYPE_COLORS.get(POKEMON_INFO[n]["types"][0],"#888") for n in POKEMON_CLASSES]
            fig = go.Figure(go.Bar(
                x=POKEMON_CLASSES, y=probs_pct, marker_color=colors,
                marker_line_color="rgba(255,255,255,0.3)", marker_line_width=1,
                text=[f"{p:.1f}%" for p in probs_pct], textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="Pokémon", yaxis_title="Probabilité (%)",
                yaxis_range=[0,118],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter,sans-serif",size=11),
                showlegend=False, height=360,
                margin=dict(t=10,b=10,l=0,r=0),
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


# ────────────────────────────────────────────────────────────────────────
# TAB 2 : CNN EXPLAINER — INTERACTIF & VULGARISE
# ────────────────────────────────────────────────────────────────────────
with tab_cnn:
    st.markdown("## 🧠 Comment fonctionne ce Pokédex ?")
    st.markdown("""
    Tu viens d'uploader une image d'un Pokémon et l'application a trouvé la réponse en une fraction de seconde.
    Mais **comment une machine «voit» et reconnaît une image ?**  
    Cette page te guide étape par étape, du pixel à la prédiction finale.
    """)

    st.divider()

    # ================================================================
    # SECTION 1 — Navigation par étape
    # ================================================================
    st.markdown("### 📍 Les 5 grandes étapes")

    STEPS = [
        "1️⃣  L’image : un tableau de chiffres",
        "2️⃣  Prétraitement & Data Augmentation",
        "3️⃣  Couches de Convolution",
        "4️⃣  Pooling : résumer l’information",
        "5️⃣  Classification finale (Dense + Softmax)",
    ]
    selected_step = st.radio(
        "Sélectionne une étape pour l’explorer :",
        STEPS, horizontal=True, label_visibility="collapsed",
    )

    st.markdown("")

    # ----------------------------------------------------------------
    # ÉTAPE 1 — L'IMAGE EST UN TABLEAU DE CHIFFRES
    # ----------------------------------------------------------------
    if selected_step == STEPS[0]:
        col_txt, col_viz = st.columns([1, 1], gap="large")
        with col_txt:
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">📷 Une image = une grille de nombres</div>
            Un ordinateur ne «voit» pas une image comme toi.  
            Il voit une <strong>matrice 3D</strong> de valeurs entre 0 et 255 :
            <ul>
              <li>🟥 Canal <strong>Rouge</strong></li>
              <li>🟩 Canal <strong>Vert</strong></li>
              <li>🟦 Canal <strong>Bleu</strong></li>
            </ul>
            Une image <strong>200×200 px</strong> = <strong>200 × 200 × 3 = 120 000 valeurs</strong>.
            <div class="metaphor">
            💡 <em>Métaphore :</em> imagine un tableau Excel de 200 lignes et 600 colonnes
            (200 par couleur). Chaque case contient un chiffre entre 0 (noir) et 255 (couleur vive).
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🔢 Normalisation</div>
            Avant de passer dans le réseau, on <strong>divise chaque valeur par 255</strong>
            pour obtenir des nombres entre 0 et 1.  
            Cela accélère et stabilise l’entraînement.
            <br><br>
            <code>pixel_normalisé = pixel / 255.0</code>
            </div>
            """, unsafe_allow_html=True)

        with col_viz:
            st.markdown("**Visualisation : valeurs RGB d’une image 8×8 simulée**")
            np.random.seed(42)
            fake_img = np.random.randint(30, 230, (8, 8, 3))
            channel = st.selectbox("Canal à visualiser", ["Rouge", "Vert", "Bleu"])
            ch_idx  = {"Rouge": 0, "Vert": 1, "Bleu": 2}[channel]
            ch_color = {"Rouge": "Reds", "Vert": "Greens", "Bleu": "Blues"}[channel]
            ch_data  = fake_img[:, :, ch_idx]
            fig_px = px.imshow(
                ch_data, color_continuous_scale=ch_color,
                text_auto=True, aspect="equal",
                labels=dict(color="Valeur (0–255)"),
            )
            fig_px.update_layout(
                title=f"Canal {channel} — valeurs brutes",
                height=320, margin=dict(t=40,b=10,l=10,r=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
            )
            fig_px.update_xaxes(showticklabels=False)
            fig_px.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_px, use_container_width=True)
            st.caption("🔎 Chaque case = un pixel. La valeur indique l'intensité de la couleur.")

    # ----------------------------------------------------------------
    # ÉTAPE 2 — PRÉTRAITEMENT & DATA AUGMENTATION
    # ----------------------------------------------------------------
    elif selected_step == STEPS[1]:
        col_txt, col_viz = st.columns([1, 1], gap="large")
        with col_txt:
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🔧 Pourquoi prétraiter ?</div>
            Le réseau attend toujours la <strong>même taille d’entrée</strong>.
            On redimensionne donc toutes les images à <strong>200×200 px</strong>
            avant l’entraînement et la prédiction.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🎲 Data Augmentation : créer de la diversité</div>
            On ne dispose que d’<strong>un seul sprite</strong> par Pokémon.
            Un modèle entraîné sur une seule image par classe <em>mémoriserait</em>
            plutôt qu’il n’<em>apprendrait</em>.
            <br><br>
            On génère donc <strong>30 variantes</strong> de chaque image :
            <ul>
              <li>🔄 Rotation aléatoire (±20°)</li>
              <li>🔍 Zoom in/out (±15%)</li>
              <li>↔️ Décalage horizontal/vertical (±10%)</li>
              <li>🔀 Flip horizontal</li>
              <li>☀️ Variation de luminosité (±15%)</li>
            </ul>
            <div class="metaphor">
            💡 <em>Métaphore :</em> si tu voulais apprendre à reconnaître un chat
            en n’ayant vu qu’une photo, tu l’inclines, la retournes, changes la lumière…
            pour ne pas rater un chat photographé différemment.
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">📂 Split train / val / test</div>
            Les images augmentées sont divisées en 3 ensembles :
            <ul>
              <li><strong>70% Train</strong> — le modèle apprend dessus</li>
              <li><strong>20% Test</strong> — évaluation finale (jamais vu pendant l’entraînement)</li>
              <li><strong>10% Validation</strong> — contrôle à chaque époque pour détecter le surapprentissage</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_viz:
            st.markdown("**Répartition des images par ensemble**")
            total_imgs = 1780
            fig_split = go.Figure(go.Pie(
                labels=["Train (70%)", "Test (20%)", "Validation (10%)"],
                values=[0.70*total_imgs, 0.20*total_imgs, 0.10*total_imgs],
                hole=0.45,
                marker_colors=["#4a90d9", "#E63946", "#2ecc71"],
                textinfo="label+percent+value",
                textfont_size=12,
            ))
            fig_split.update_layout(
                height=280, margin=dict(t=20,b=20,l=0,r=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_split, use_container_width=True)

            st.markdown("**Nombre d’images par Pokémon (après augmentation)**")
            aug_counts = {"Brut (1 sprite)": 1, "Après augmentation (×30)": 30,
                          "Train (~70%)": 21, "Val (~10%)": 3, "Test (~20%)": 6}
            fig_aug = go.Figure(go.Bar(
                x=list(aug_counts.keys()), y=list(aug_counts.values()),
                marker_color=["#ccc", "#4a90d9", "#2ecc71", "#FFC107", "#E63946"],
                text=list(aug_counts.values()), textposition="outside",
            ))
            fig_aug.update_layout(
                height=260, margin=dict(t=10,b=10,l=0,r=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11), showlegend=False,
                yaxis_range=[0, 36],
            )
            fig_aug.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_aug, use_container_width=True)

    # ----------------------------------------------------------------
    # ÉTAPE 3 — CONVOLUTION
    # ----------------------------------------------------------------
    elif selected_step == STEPS[2]:
        col_txt, col_viz = st.columns([1, 1], gap="large")
        with col_txt:
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🧱 Une couche de Convolution — c’est quoi ?</div>
            C’est le cœur d’un CNN. Elle <strong>scanne l’image avec un petit filtre</strong>
            (ici 4×4 pixels) qui se déplace sur toute l’image.
            <br><br>
            À chaque position, le filtre <strong>multiplie ses valeurs</strong> par les pixels
            correspondants, additionne tout, et produit <strong>une seule valeur de sortie</strong>.
            L’ensemble des résultats forme une <strong>feature map</strong> — une carte de «caractéristiques».
            <div class="metaphor">
            💡 <em>Métaphore :</em> c’est comme passer un pochoir sur une image et noter
            à quel point chaque zone ressemble au pochoir.
            Un pochoir «contour» répondra fort là où il y a des bords.
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🧠 Ce que chaque couche détecte</div>
            <ul>
              <li><strong>Conv 128 filtres</strong> — bords, couleurs, gradients basiques</li>
              <li><strong>Conv 64 filtres</strong> — formes simples, textures</li>
              <li><strong>Conv 32 filtres</strong> — parties du corps, silhouettes</li>
              <li><strong>Conv 16 filtres</strong> — caractéristiques de haut niveau</li>
            </ul>
            <div class="metaphor">
            💡 <em>Métaphore :</em> la première couche voit les «briques»,
            les suivantes assemblent ces briques en «murs», puis en «batiment».
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">⚡ Activation ReLU</div>
            Après chaque convolution, on applique la fonction <strong>ReLU</strong> :
            <br><code>f(x) = max(0, x)</code>
            <br><br>
            Elle <strong>élimine les valeurs négatives</strong> (réponses «non pertinentes»)
            et accélère l’apprentissage en gardant le réseau non-linéaire.
            </div>
            """, unsafe_allow_html=True)

        with col_viz:
            st.markdown("**Simulation interactive : applique un filtre sur une image**")
            filter_type = st.selectbox(
                "Choisir un filtre à simuler :",
                ["Détection de contours (horizontal)",
                 "Détection de contours (vertical)",
                 "Flou (moyenne)",
                 "Netteté (sharpening)"],
            )
            filters = {
                "Détection de contours (horizontal)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
                "Détection de contours (vertical)":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
                "Flou (moyenne)":                      np.ones((3,3))/9,
                "Netteté (sharpening)":               np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
            }
            sel_filter = filters[filter_type]

            # Générer une image de test simple (gradient + forme)
            np.random.seed(7)
            base = np.zeros((12,12))
            base[3:9, 3:9] = 200   # carré blanc
            base += np.random.randint(0, 30, (12,12))  # bruit léger
            base = np.clip(base, 0, 255)

            # Convolution manuelle (valid)
            kh, kw = sel_filter.shape
            H, W = base.shape
            out_h, out_w = H - kh + 1, W - kw + 1
            out = np.zeros((out_h, out_w))
            for i in range(out_h):
                for j in range(out_w):
                    out[i,j] = np.sum(base[i:i+kh, j:j+kw] * sel_filter)
            out_relu = np.maximum(out, 0)

            col_f1, col_f2, col_f3 = st.columns(3)
            def mini_heatmap(data, title, colorscale="Greys"):
                f = px.imshow(data, color_continuous_scale=colorscale,
                              aspect="equal", text_auto=False)
                f.update_layout(title=dict(text=title,font=dict(size=11)),
                                height=200, margin=dict(t=30,b=0,l=0,r=0),
                                coloraxis_showscale=False,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)")
                f.update_xaxes(showticklabels=False)
                f.update_yaxes(showticklabels=False)
                return f

            with col_f1:
                st.plotly_chart(mini_heatmap(base, "Image d’entrée"), use_container_width=True)
            with col_f2:
                st.plotly_chart(mini_heatmap(out, "Après convolution", "RdBu"), use_container_width=True)
            with col_f3:
                st.plotly_chart(mini_heatmap(out_relu, "Après ReLU", "Reds"), use_container_width=True)

            st.caption("🔎 Les zones rouges après ReLU = là où le filtre a «répondu» fort.")

            st.markdown("**Filtre appliqué (3×3)**")
            fig_filt = px.imshow(
                sel_filter, color_continuous_scale="RdBu", text_auto=True,
                aspect="equal",
            )
            fig_filt.update_layout(
                height=160, margin=dict(t=10,b=0,l=0,r=0),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_filt.update_xaxes(showticklabels=False)
            fig_filt.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_filt, use_container_width=True)

    # ----------------------------------------------------------------
    # ÉTAPE 4 — POOLING
    # ----------------------------------------------------------------
    elif selected_step == STEPS[3]:
        col_txt, col_viz = st.columns([1, 1], gap="large")
        with col_txt:
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">↓ MaxPooling — réduire sans perdre l’essentiel</div>
            Après chaque convolution, la feature map est encore grande.
            Le <strong>MaxPooling 2×2</strong> la <strong>divise par 2 en taille</strong>
            en gardant uniquement la <strong>valeur maximale</strong> de chaque bloc 2×2.
            <div class="metaphor">
            💡 <em>Métaphore :</em> imagine que tu résumes un texte en gardant
            uniquement le mot le plus important de chaque paragraphe.
            Tu perds les détails mais tu gardes l’essentiel.
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">📉 Évolution de la taille au fil des couches</div>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
              <tr style="background:#f0f4ff;font-weight:700">
                <td style="padding:6px">Couche</td>
                <td style="padding:6px">Taille sortie</td>
                <td style="padding:6px">Params (approx.)</td>
              </tr>
              <tr><td style="padding:5px">Entrée</td><td>200×200×3</td><td>—</td></tr>
              <tr style="background:#fff0f0"><td style="padding:5px">Conv 128</td><td>197×197×128</td><td>6 272</td></tr>
              <tr style="background:#f0fff4"><td style="padding:5px">Pool</td><td>98×98×128</td><td>—</td></tr>
              <tr style="background:#fff0f0"><td style="padding:5px">Conv 64</td><td>95×95×64</td><td>131 136</td></tr>
              <tr style="background:#f0fff4"><td style="padding:5px">Pool</td><td>47×47×64</td><td>—</td></tr>
              <tr style="background:#fff0f0"><td style="padding:5px">Conv 32</td><td>44×44×32</td><td>32 800</td></tr>
              <tr style="background:#f0fff4"><td style="padding:5px">Pool</td><td>22×22×32</td><td>—</td></tr>
              <tr style="background:#fff0f0"><td style="padding:5px">Conv 16</td><td>19×19×16</td><td>8 208</td></tr>
              <tr style="background:#f0fff4"><td style="padding:5px">Pool</td><td>9×9×16</td><td>—</td></tr>
              <tr style="background:#f0f4ff"><td style="padding:5px">Flatten</td><td>1296</td><td>—</td></tr>
              <tr style="background:#f0f4ff"><td style="padding:5px">Dense 64</td><td>64</td><td>83 008</td></tr>
              <tr style="background:linear-gradient(135deg,#fff0f8,#f0f4ff)"><td style="padding:5px">Dense 10</td><td>10</td><td>650</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with col_viz:
            st.markdown("**Simulation MaxPooling 2×2**")
            np.random.seed(3)
            inp4 = np.random.randint(0, 100, (8, 8))
            # MaxPooling 2x2
            out4 = np.array([
                [np.max(inp4[i*2:(i+1)*2, j*2:(j+1)*2]) for j in range(4)]
                for i in range(4)
            ])
            c1, c2 = st.columns(2)
            with c1:
                fig_in = px.imshow(inp4, text_auto=True, color_continuous_scale="Blues", aspect="equal")
                fig_in.update_layout(title="Entrée (8×8)", height=260,
                                     margin=dict(t=30,b=0,l=0,r=0), coloraxis_showscale=False,
                                     plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                fig_in.update_xaxes(showticklabels=False)
                fig_in.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_in, use_container_width=True)
                st.caption("↑ Feature map avant pooling")
            with c2:
                fig_out = px.imshow(out4, text_auto=True, color_continuous_scale="Reds", aspect="equal")
                fig_out.update_layout(title="Sortie MaxPool (4×4)", height=260,
                                      margin=dict(t=30,b=0,l=0,r=0), coloraxis_showscale=False,
                                      plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                fig_out.update_xaxes(showticklabels=False)
                fig_out.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_out, use_container_width=True)
                st.caption("↓ Après pooling : moitié plus petite, valeurs max conservées")

            st.markdown("**Réduction de taille au fil des 4 blocs**")
            sizes = [200, 98, 47, 22, 9]
            labels = ["Entrée", "Après Pool 1", "Après Pool 2", "Après Pool 3", "Après Pool 4"]
            fig_sz = go.Figure(go.Bar(
                x=labels, y=sizes,
                marker_color=["#ccc","#4a90d9","#4a90d9","#4a90d9","#4a90d9"],
                text=[f"{s}×{s}" for s in sizes], textposition="outside",
            ))
            fig_sz.update_layout(
                height=240, yaxis_range=[0, 240],
                margin=dict(t=10,b=0,l=0,r=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11), showlegend=False,
            )
            fig_sz.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_sz, use_container_width=True)

    # ----------------------------------------------------------------
    # ÉTAPE 5 — DENSE + SOFTMAX
    # ----------------------------------------------------------------
    elif selected_step == STEPS[4]:
        col_txt, col_viz = st.columns([1, 1], gap="large")
        with col_txt:
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">📏 Flatten : aplatir en 1D</div>
            Après les 4 blocs Conv+Pool, on obtient un volume <strong>9×9×16 = 1296 valeurs</strong>.
            La couche <strong>Flatten</strong> les «empile» en un seul vecteur de 1296 nombres.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🔗 Dense 64 — combiner les caractéristiques</div>
            Une couche <strong>fully connected (Dense)</strong> connecte
            <em>chaque</em> entrée à <em>chaque</em> sortie.
            Ici : 1296 entrées → 64 neurones.<br><br>
            Chaque neurone calcule une somme pondérée de toutes les entrées,
            puis applique ReLU. C’est là que le modèle <strong>«assemble»
            les pièces du puzzle</strong> pour former une opinion globale.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            <div class="concept-box">
            <div class="concept-title">🎯 Softmax — la décision finale</div>
            La dernière couche Dense a <strong>10 neurones</strong> — un par Pokémon.
            La fonction <strong>Softmax</strong> convertit ces scores bruts en
            <strong>probabilités qui somment à 100%</strong>.
            <br><br>
            La classe avec la <strong>probabilité la plus élevée</strong> est la prédiction.
            <div class="metaphor">
            💡 <em>Métaphore :</em> après avoir tout examiné, le réseau dit :
            « je suis sûr à 87% que c’est Pikachu, 8% Eevee, 5% autre… »
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col_viz:
            st.markdown("**Simulation : scores bruts → Softmax**")
            np.random.seed(42)
            raw_scores = np.array([0.3, 0.5, 0.1, 0.2, 0.4, 0.0, 0.6, 2.8, 0.3, 0.2])
            # softmax
            e = np.exp(raw_scores - raw_scores.max())
            softmax_out = e / e.sum()

            colors_bar = [TYPE_COLORS.get(POKEMON_INFO[n]["types"][0], "#888") for n in POKEMON_CLASSES]

            fig_raw = go.Figure(go.Bar(
                x=POKEMON_CLASSES, y=raw_scores,
                marker_color=colors_bar,
                text=[f"{v:.1f}" for v in raw_scores], textposition="outside",
            ))
            fig_raw.update_layout(
                title="Scores bruts (logits)",
                height=220, margin=dict(t=40,b=0,l=0,r=0), yaxis_range=[-0.5,3.5],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=10), showlegend=False,
            )
            fig_raw.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_raw, use_container_width=True)

            fig_sm = go.Figure(go.Bar(
                x=POKEMON_CLASSES, y=softmax_out*100,
                marker_color=colors_bar,
                text=[f"{v*100:.1f}%" for v in softmax_out], textposition="outside",
            ))
            fig_sm.update_layout(
                title="Après Softmax (probabilités %)",
                height=220, margin=dict(t=40,b=0,l=0,r=0), yaxis_range=[0,75],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=10), showlegend=False,
            )
            fig_sm.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_sm, use_container_width=True)
            st.caption("🔎 Mewtwo (🔮) a le score le plus élevé → c’est la prédiction.")

    st.divider()

    # ================================================================
    # SECTION 2 — Pipeline complet (toujours visible)
    # ================================================================
    st.markdown("### 🖥️ Vue d’ensemble du pipeline")
    st.caption("Chaque bloc représente une étape de la transformation de l’image jusqu’à la prédiction.")

    pipeline_layers = [
        ("step-input",  "📷", "Image brute",             "Sprite PNG, taille variable"),
        ("step-aug",    "🔧", "Prétraitement",          "Redim. 200×200 · Normalisation ÷255"),
        ("step-aug",    "🎲", "Data Augmentation",      "×30 variantes : rotation, zoom, flip, lumière"),
        ("step-aug",    "📂", "Split 70/20/10",         "Train · Test · Validation"),
        ("step-conv",   "🧱", "Conv2D 128 filtres · ReLU", "Détecte bords, couleurs, gradients"),
        ("step-pool",   "↓",  "MaxPooling 2×2",        "200×200 → 98×98"),
        ("step-conv",   "🧱", "Conv2D 64 filtres · ReLU",  "Détecte formes, textures"),
        ("step-pool",   "↓",  "MaxPooling 2×2",        "98×98 → 47×47"),
        ("step-conv",   "🧱", "Conv2D 32 filtres · ReLU",  "Détecte silhouettes"),
        ("step-pool",   "↓",  "MaxPooling 2×2",        "47×47 → 22×22"),
        ("step-conv",   "🧱", "Conv2D 16 filtres · ReLU",  "Caractéristiques haut niveau"),
        ("step-pool",   "↓",  "MaxPooling 2×2",        "22×22 → 9×9"),
        ("step-dense",  "📏", "Flatten",                "9×9×16 = 1296 valeurs → vecteur 1D"),
        ("step-dense",  "🔗", "Dense 64 · ReLU",        "1296 → 64 neurones"),
        ("step-out",    "🎯", "Dense 10 · Softmax",     "64 → 10 probabilités (une par Pokémon)"),
    ]

    cols_pipe = st.columns(5)
    for idx, (cls, icon, name, desc) in enumerate(pipeline_layers):
        with cols_pipe[idx % 5]:
            st.markdown(
                f"<div class='step-card {cls}' style='min-height:90px'>"
                f"<div style='font-size:1.4rem'>{icon}</div>"
                f"<div class='step-title' style='font-size:0.82rem'>{name}</div>"
                f"<div class='step-sub' style='font-size:0.73rem'>{desc}</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────
# TAB 3 : DATASET & PROJET
# ────────────────────────────────────────────────────────────────────────
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
                f'padding:1px 8px;border-radius:10px;font-size:0.78rem;margin:2px">{t}</span>'
                for t in info["types"]
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
        SELECTED = ['pikachu', 'charizard', 'mewtwo', ...]
        N_AUGMENTED = 50   # images par Pokémon
        EPOCHS = 20
        ```
        Puis sauvegardez :
        ```python
        model.save('pokemon_cnn_model.keras')
        ```
        """)
        st.markdown("### 🗂️ Structure du repo")
        st.code("""
pokedexCNN/
├── app.py
├── pokemon_info.py
├── pokemon_cnn_model.keras
├── pokemon.csv
├── pokemam_10_epoch.ipynb
├── requirements.txt
└── .streamlit/config.toml
        """, language="")
        st.markdown("### 🔗 Liens utiles")
        st.link_button("📓 Voir le notebook", "https://github.com/BadreddineEK/pokedexCNN/blob/main/pokemam_10_epoch.ipynb")
        st.link_button("⭐ GitHub du projet", "https://github.com/BadreddineEK/pokedexCNN")

# ═══ FOOTER ════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;color:#bbb;font-size:0.82rem'>"
    "Projet école Polytech Lyon 2022–2023 — repris &amp; modernizé · "
    "<a href='https://github.com/BadreddineEK/pokedexCNN' style='color:#bbb'>GitHub</a> · "
    "<a href='https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types' style='color:#bbb'>Dataset Kaggle</a></p>",
    unsafe_allow_html=True,
)
