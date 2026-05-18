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
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

#CONFIGURACIÓN STREAMLIT
st.set_page_config(
    page_title="Moodify AI",
    page_icon="🎵",
    layout="wide"
)

#CSS
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
    color: #111827;
}
.stTextInput input {
    border-radius: 12px;
}
.emotion-box {
    padding: 15px;
    border-radius: 10px;
    background: #e5e7eb;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

#SPOTIFY
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

def preprocess(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


class EmotionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc_out(x)

vectorizer = joblib.load("models/vectorizer.pkl")
id2label = joblib.load("models/id2label.pkl")

input_dim = len(vectorizer.get_feature_names_out())
model = EmotionMLP(input_dim, 128, len(id2label))
model.load_state_dict(torch.load("models/emociones_mlp.pth"))
model.eval()

emotion_emojis = {
    "alegria": "😄",
    "tristeza": "😢",
    "enojo": "😡",
    "calma": "😌",
    "estres": "😰"
}

emotion_to_keywords = {
    "alegria": ["happy", "party", "dance"],
    "tristeza": ["sad", "melancholy", "acoustic"],
    "enojo": ["rock", "metal", "rage"],
    "calma": ["relax", "chill", "ambient"],
    "estres": ["focus", "study", "lofi"]
}

st.title("🎵 Moodify AI")
st.subheader("Recomendador musical basado en emociones")

texto = st.text_input("¿Cómo te sientes hoy?")

if texto:
    processed = preprocess(texto)
    features = vectorizer.transform([processed]).toarray()
    features_tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred = torch.argmax(outputs, dim=1).item()

    emocion = id2label[pred]
    confianza = probs[pred]
    emoji = emotion_emojis.get(emocion, "🎵")

    st.markdown(
        f"""
        <div class='emotion-box'>
            <h3>{emoji} Emoción detectada: {emocion}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"Confianza del modelo: {confianza:.2%}")
    st.progress(float(confianza))

    st.subheader("🎧 Playlists recomendadas")
    keywords = emotion_to_keywords.get(emocion, [emocion])
    for kw in keywords:
        playlists = sp.search(q=kw, type="playlist", limit=2)
        if playlists and playlists.get("playlists"):
            for p in playlists["playlists"]["items"]:
                if p:
                    st.markdown(f"""
                    **{p['name']}**  
                    🔗 {p['external_urls']['spotify']}
                    """)

    st.subheader("🎶 Canciones sugeridas")
    for kw in keywords:
        results = sp.search(q=kw, type="track", limit=3)
        if results and results.get("tracks"):
            for track in results["tracks"]["items"]:
                st.write(f"{track['name']} - {track['artists'][0]['name']}")
                preview = track.get("preview_url")
                if preview:
                    st.audio(preview)
                else:
                    st.write("⚠️ No hay preview disponible")


st.sidebar.title("📊 Métricas del Modelo")

df = pd.read_csv("data/emociones.csv")
df['texto'] = df['texto'].apply(preprocess)

y = pd.Series(df['emocion']).astype('category')
labels = list(y.cat.categories)
y_codes = y.cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    vectorizer.transform(df['texto']).toarray(),
    y_codes,
    test_size=0.2,
    random_state=42,
    stratify=y_codes
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1).numpy()

report = classification_report(y_test, preds, target_names=labels, output_dict=True)

for label, metrics in report.items():
    if isinstance(metrics, dict):
        st.sidebar.write(
            f"**{label}** → Precisión: {metrics['precision']:.2f}, "
            f"Recall: {metrics['recall']:.2f}, "
            f"F1: {metrics['f1-score']:.2f}"
        )
