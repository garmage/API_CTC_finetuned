FROM python:3.8-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances et installer les packages nécessaires
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . /app

# Exposer le port sur lequel Flask va écouter
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
