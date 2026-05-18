# Moodify AI - Recomendador Musical Basado en Emociones

Moodify AI es un prototipo funcional (MVP) diseñado para la optimización del bienestar emocional a través de la música. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y Deep Learning para analizar las expresiones textuales de los usuarios sobre su estado de ánimo, clasificar su emoción predominante mediante una red neuronal MLP y generar recomendaciones automatizadas de playlists y canciones consumidas directamente desde la API oficial de Spotify.

## Características Principales
- **Detección de Emociones:** Clasificación de texto en 5 categorías emocionales clave: Alegría, Tristeza, Enojo, Calma y Estrés mediante una red neuronal MLP entrenada con PyTorch.
- **Análisis de Confianza:** Distribución probabilística vía Softmax con barra de progreso interactiva que muestra la certeza de la predicción del modelo.
- **Integración con Spotify API:** Búsqueda dinámica y en tiempo real de playlists y pistas musicales que correspondan al estado de ánimo detectado mediante palabras clave por emoción.
- **Reproducción Integrada:** Audio previews de 30 segundos embebidos directamente en la interfaz de usuario cuando están disponibles en la plataforma.
- **Métricas del Modelo:** Panel lateral con precisión, recall y F1-score por clase calculados en tiempo de ejecución sobre el conjunto de prueba.

---

## Arquitectura y Stack Tecnológico

El proyecto está construido bajo una arquitectura monolítica desacoplada en su procesamiento de datos, dividida en la fase de entrenamiento local y la fase de inferencia/servido en la aplicación web:

- **Frontend / Interfaz:** Framework ágil para despliegue de aplicaciones de datos con soporte para audio embebido y métricas en sidebar.
- **Modelado de IA & NLP:**
  -  PyTorch — Red neuronal MLP (EmotionMLP) para clasificación multiclase de emociones.
  - Scikit — Vectorización TF-IDF (max_features=5000) y serialización de artefactos.
  - NLTK — para la limpieza de Stopwords en español.
- **Integración Externa:** Spotipy con flujo SpotifyOAuth para búsqueda de playlists y tracks por palabras clave emocionales.
- **Gestión de credenciales:** python-dotenv para carga de variables de entorno desde .env.

---

## Estructura del Repositorio

```text
SEMINARIOIA/
│
├── data/
│   └── emociones.csv           # Dataset con 325 muestras etiquetadas (65 por emoción)
│
├── models/                     # Artefactos generados por train.py (ignorado por Git)
│   ├── emociones_mlp.pth       # Pesos del modelo MLP entrenado (PyTorch state_dict)
│   ├── vectorizer.pkl          # Transformador TF-IDF serializado con Joblib
│   └── id2label.pkl            # Diccionario de mapeo índice → etiqueta de emoción
│
├── reports/
│   └── confusion_matrix.png    # Matriz de confusión generada tras el entrenamiento
│
├── app.py                      # Aplicación principal e interfaz gráfica en Streamlit
├── train.py                    # Script de entrenamiento del modelo MLP y generación de reportes
├── requirements.txt            # Definición de dependencias del entorno de ejecución
├── .env                        # Variables de entorno con credenciales de Spotify (no versionado)
└── .gitignore                  # Exclusión de archivos pesados o sensibles
```

## Instalación y Configuración Local

1. Prerrequisitos

   Asegúrate de contar con Python 3.10+ instalado en tu sistema.

3. Clonar el repositorio

   Bash

   git clone [https://github.com/JonyGarciaaa/SeminarioIA.git](https://github.com/JonyGarciaaa/SeminarioIA.git)

   cd SeminarioIA

4. Instalar Dependencias

   Se recomienda utilizar un entorno virtual (venv):

   Bash

   python -m venv venv

   source venv/bin/activate

   En Windows:

   venv\Scripts\activate

   pip install -r requirements.txt

6. Configurar Variables de Entorno

   El proyecto requiere interactuar con la API de Spotify. Para la ejecución local, asegúrate de que exista un archivo `.env` en la raíz del proyecto con la siguiente estructura (puedes guiarte de `.env.example`):

   ```env
    SPOTIFY_CLIENT_ID=tu_client_id_de_spotify
    SPOTIFY_CLIENT_SECRET=tu_client_secret_de_spotify
    SPOTIFY_REDIRECT_URI=http://localhost:8501
    ```

8. Entrenar el Modelo

   Antes de ejecutar la app, procesa el dataset ejecutando el script de entrenamiento:

   Bash

   python train.py

10. Ejecutar la Aplicación

    Bash

    streamlit run app.py

La aplicación se abrirá automáticamente en tu navegador web en la dirección http://localhost:8501.
