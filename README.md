# Moodify AI - Recomendador Musical Basado en Emociones

Moodify AI es un prototipo funcional (MVP) diseñado para la optimización del bienestar emocional a través de la música. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning para analizar las expresiones textuales de los usuarios sobre su estado de ánimo, clasificar su emoción predominante y generar recomendaciones automatizadas de playlists y canciones consumidas directamente desde la API oficial de Spotify.

## Características Principales
- **Detección de Emociones:** Clasificación de texto en 5 categorías emocionales clave: Alegría, Tristeza, Enojo, Calma y Estrés.
- **Análisis de Confianza:** Desglose probabilístico mediante gráficos interactivos sobre la certeza de la predicción del modelo.
- **Integración con Spotify API:** Búsqueda dinámica y en tiempo real de playlists y pistas musicales que correspondan al estado de ánimo detectado.
- **Reproducción Integrada:** Audio previews embebidos directamente en la interfaz de usuario cuando están disponibles en la plataforma.

---

## Arquitectura y Stack Tecnológico

El proyecto está construido bajo una arquitectura monolítica desacoplada en su procesamiento de datos, dividida en la fase de entrenamiento local y la fase de inferencia/servido en la aplicación web:

- **Frontend / Interfaz:** Streamlit (v1.33.0) - Framework ágil para despliegue de aplicaciones de datos.
- **Modelado de IA & NLP:** - Scikit-learn & Joblib para la vectorización (TF-IDF) y el clasificador clásico.
  - NLTK para la limpieza de Stopwords en español.
- **Integración Externa:** Spotipy (Wrapper oficial de Spotify Web API).

---

## Estructura del Repositorio

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
```

Instalación y Configuración Local
1. Prerrequisitos
Asegúrate de contar con Python 3.10+ instalado en tu sistema.

2. Clonar el repositorio
Bash
git clone [https://github.com/JonyGarciaaa/SeminarioIA.git](https://github.com/JonyGarciaaa/SeminarioIA.git)
cd SeminarioIA
3. Instalar Dependencias
Se recomienda utilizar un entorno virtual (venv):

Bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
4. Configurar Variables de Entorno
Crea un archivo .env en la raíz del proyecto basándote en .env.example:

Fragmento de código
SPOTIFY_CLIENT_ID=tu_client_id_de_spotify_developer
SPOTIFY_CLIENT_SECRET=tu_client_secret_de_spotify_developer
SPOTIFY_REDIRECT_URI=http://localhost:8501
Nota: Debes registrar una aplicación en el Spotify Developer Dashboard para obtener estas credenciales.

5. Entrenar el Modelo
Antes de ejecutar la app, procesa el dataset ejecutando el script de entrenamiento:

Bash
python train.py
6. Ejecutar la Aplicación
Bash
streamlit run app.py
La aplicación se abrirá automáticamente en tu navegador web en la dirección http://localhost:8501.
