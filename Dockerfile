# ---- Étape 1 : Base Python ----
FROM python:3.11-slim

# Empêche Python d’écrire des fichiers .pyc et d’utiliser le buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Étape 2 : Dossier de travail ----
WORKDIR /app

# ---- Étape 3 : Dépendances ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Étape 4 : Copier le code ----
COPY . .

# ---- Étape 5 : Variables d’environnement utiles ----
ENV MLFLOW_TRACKING_URI=mlruns
ENV PORT=8501

# ---- Étape 6 : Exposer le port de Streamlit ----
EXPOSE 8501

# ---- Étape 7 : Commande de lancement ----
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]