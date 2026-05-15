import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Descargar stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- Preprocesamiento ---
def preprocess(text):
    text = text.lower()

    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = [w for w in text.split() if w not in stop_words]

    return " ".join(tokens)

# --- Cargar dataset ---
df = pd.read_csv("data/emociones.csv")
df['texto'] = df['texto'].apply(preprocess)

print(df['emocion'].value_counts())

# --- Vectorización ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto'])

y = pd.Series(df['emocion']).astype('category')
labels = list(y.cat.categories)
y_codes = y.cat.codes

id2label = dict(enumerate(labels))

# Guardar vectorizador y etiquetas
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(id2label, "models/id2label.pkl")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_codes,
    test_size=0.2,
    random_state=42,
    stratify=y_codes
)

# --- Modelo ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Predicciones ---
preds = model.predict(X_test)

# --- Métricas ---
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, preds, target_names=labels))

# --- Matriz de confusión ---
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.savefig("reports/confusion_matrix.png")
# --- Guardar modelo ---
joblib.dump(model, "models/model.pkl")

print("✅ Modelo guardado correctamente")