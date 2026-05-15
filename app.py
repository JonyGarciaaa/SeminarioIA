import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
import joblib
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords

# =========================
# NLTK
# =========================
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# =========================
# CONFIGURACIÓN STREAMLIT
# =========================
st.set_page_config(
    page_title="Moodify AI",
    page_icon="🎵",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stTextInput input {
    border-radius: 12px;
}

.emotion-box {
    padding: 20px;
    border-radius: 15px;
    background: #1e293b;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SPOTIFY
# =========================
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="playlist-read-private"
    )
)

# =========================
# PREPROCESAMIENTO
# =========================
def preprocess(text):

    text = text.lower()

    # quitar acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # quitar caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # eliminar stopwords
    tokens = [w for w in text.split() if w not in stop_words]

    return " ".join(tokens)

# =========================
# CARGAR MODELO
# =========================
vectorizer = joblib.load("models/vectorizer.pkl")
id2label = joblib.load("models/id2label.pkl")
model = joblib.load("models/model.pkl")

# =========================
# EMOJIS
# =========================
emotion_emojis = {
    "alegria": "😄",
    "tristeza": "😢",
    "enojo": "😡",
    "calma": "😌",
    "estres": "😰"
}

# =========================
# KEYWORDS SPOTIFY
# =========================
emotion_to_keywords = {
    "alegria": ["happy", "party", "dance"],
    "tristeza": ["sad", "melancholy", "acoustic"],
    "enojo": ["rock", "metal", "rage"],
    "calma": ["relax", "chill", "ambient"],
    "estres": ["focus", "study", "lofi"]
}

# =========================
# UI PRINCIPAL
# =========================
st.title("🎵 Moodify AI")
st.subheader("Recomendador musical basado en emociones")

texto = st.text_input("¿Cómo te sientes hoy?")

if texto:

    # =========================
    # PREDECIR
    # =========================
    processed = preprocess(texto)

    features = vectorizer.transform([processed])

    pred = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    emocion = id2label[pred]

    confianza = max(probabilities)

    emoji = emotion_emojis.get(emocion, "🎵")

    # =========================
    # RESULTADO
    # =========================
    st.markdown(
        f"""
        <div class='emotion-box'>
            <h1>{emoji}</h1>
            <h2>Emoción detectada: {emocion}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"Confianza del modelo: {confianza:.2%}")

    st.progress(float(confianza))

    # =========================
    # PROBABILIDADES
    # =========================
    st.subheader("📊 Probabilidades")

    prob_df = pd.DataFrame({
        "Emoción": [id2label[i] for i in range(len(probabilities))],
        "Probabilidad": probabilities
    })

    st.bar_chart(prob_df.set_index("Emoción"))

    # =========================
    # PLAYLISTS
    # =========================
    st.subheader("🎧 Playlists recomendadas")

    keywords = emotion_to_keywords.get(emocion, [emocion])

    for kw in keywords:

        playlists = sp.search(
            q=kw,
            type="playlist",
            limit=2
        )

        if playlists and playlists.get("playlists"):

            for p in playlists["playlists"]["items"]:

                if p:

                    st.markdown(f"""
                    ### {p['name']}
                    🔗 {p['external_urls']['spotify']}
                    """)

    # =========================
    # CANCIONES
    # =========================
    st.subheader("🎶 Canciones sugeridas")

    for kw in keywords:

        results = sp.search(
            q=kw,
            type="track",
            limit=3
        )

        if results and results.get("tracks"):

            for track in results["tracks"]["items"]:

                st.write(
                    f"{track['name']} - {track['artists'][0]['name']}"
                )

                preview = track.get("preview_url")

                if preview:
                    st.audio(preview)
                else:
                    st.write("⚠️ No hay preview disponible")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("ℹ️ Información")

st.sidebar.write("""
Moodify AI utiliza:

- NLP
- TF-IDF
- Machine Learning
- Spotify API
- Scikit-learn

para detectar emociones y recomendar música.
""")