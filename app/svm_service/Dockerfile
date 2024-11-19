# Utiliser une image officielle Python comme image de base
FROM python:3.8-buster

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances dans le conteneur
COPY requirements.txt .

# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu de votre dossier svm_service dans le conteneur
COPY . .

# Créer le dossier pour stocker les fichiers téléchargés si ce n'est pas déjà fait
RUN mkdir -p /app/Test_Music

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Exposer le port que Streamlit utilise
EXPOSE 8501

# Commande pour lancer l'application Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
