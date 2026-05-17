# 🎵 Moodify AI - Recomendador Musical Basado en Emociones

Moodify AI es un prototipo funcional (MVP) diseñado para la optimización del bienestar emocional a través de la música. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning para analizar las expresiones textuales de los usuarios sobre su estado de ánimo, clasificar su emoción predominante y generar recomendaciones automatizadas de playlists y canciones consumidas directamente desde la API oficial de Spotify.

## 🚀 Características Principales
- **Detección de Emociones:** Clasificación de texto en 5 categorías emocionales clave: Alegría, Tristeza, Enojo, Calma y Estrés.
- **Análisis de Confianza:** Desglose probabilístico mediante gráficos interactivos sobre la certeza de la predicción del modelo.
- **Integración con Spotify API:** Búsqueda dinámica y en tiempo real de playlists y pistas musicales que correspondan al estado de ánimo detectado.
- **Reproducción Integrada:** Audio previews embebidos directamente en la interfaz de usuario cuando están disponibles en la plataforma.

---

## 🛠️ Arquitectura y Stack Tecnológico

El proyecto está construido bajo una arquitectura monolítica desacoplada en su procesamiento de datos, dividida en la fase de entrenamiento local y la fase de inferencia/servido en la aplicación web:

- **Frontend / Interfaz:** Streamlit (v1.33.0) - Framework ágil para despliegue de aplicaciones de datos.
- **Modelado de IA & NLP:** - Scikit-learn & Joblib para la vectorización (TF-IDF) y el clasificador clásico.
  - NLTK para la limpieza de Stopwords en español.
- **Integración Externa:** Spotipy (Wrapper oficial de Spotify Web API).

---

## 📂 Estructura del Repositorio

```text
SEMINARIOIA/
│
├── .idea/                  # Configuraciones del IDE (PyCharm)
├── data/
│   └── emociones.csv       # Dataset local con muestras etiquetadas para el entrenamiento
├── models/                 # Modelos y serializaciones generadas por train.py (Ignorado por Git)
│   ├── vectorizer.pkl      # Vocabulario/Transformador TF-IDF
│   ├── id2label.pkl        # Diccionario de mapeo de índices a emociones
│   └── model.pkl           # Modelo clasificador entrenado
│
├── app.py                  # Aplicación principal e interfaz gráfica en Streamlit
├── train.py                # Script de automatización de entrenamiento del modelo
├── requirements.txt        # Definición de dependencias del entorno de ejecución
├── .env.example            # Plantilla de variables de entorno para las credenciales de Spotify
└── .gitignore              # Configuración de exclusión de archivos pesados o sensibles
