## Agora-NLP

Outil d'analyse de réponses ouvertes pour le projet Agora.

## Structure du répertoire

- **agora_topic_modeling** : dossier contenant le code l'analyse de texte
  - **code** : dossier du code des pages
  - **dags** : dossier de code utilisé par les page notamment pour récupérer des données ou faire du traitement de données avant leur affichage
  -*.ipynb : Notebook contenant du code pouvant servir à l'exécution des pipelines de manière manuelle 
  - **webapp** : dossier contenant le code de la webapp utilisé pour orchestrer les analyses de données
- **Procfile** : fichier utile au déploiement sur Scalingo
- **requirement.txt** : fichier contenant les dépendances des librairies pythons
- *.env.template* : fichier template des variables d'environnement utilisés pour le projet


## Pré-requis

- Avoir installé Python

## Installation

L'installation est un processus en 3 étapes :
- Crée un nouvel **environement virtuel** *python* (*aka venv*)
  - Cette étape est seuelement nécessaire la première fois que vous créez un *venv*
- Activer **l'environnement virtuel** *python*
- Installer les dépendances requises

```sh
# Create a new environment named as you would like
python3 -m venv "venv"
```

```sh
# Load the virtual environment you created previously
source venv/bin/activate
```

```sh
# Install the required dependencies that have been stored in a file for the occasion
pip install -r install/requirements.txt
```

### Variables d'environnements

- **`TOPIC_THRESHOLD_FOR_SUBTOPIC`**: Valeur en pourcentage utilisés pour savoir à partir de quel représentation total des données par un topic est-ce qu'on calcul ses sous-topics. Exemple si `TOPIC_THRESHOLD_FOR_SUBTOPIC=5` alors on calcul les sous-topics pour tous les topics représentant au moins 5% des données de réponses.
- **`AGORA_NLP_URL_INSERT`**: URL *sqlalchemy* pour insérer les données analysées dans la base de données *Agora-nlp*
- **`AGORA_PROD_URL`**: URL *sqlalchemy* pour lire les données depuis la base de données *Agora-prod*


### Déploiement
Utilisation de la plateforme Scalingo pour déployer cet outil de gestion de pipeline.
Déploiement manuel de l'application sur l'interface de Scalingo en sélectionnant la branche à déployer.
Scalingo utilise un *Procfile* situé à la racine du projet pour savoir quoi lancer au démarrage de l'application déployée.

### Usage

```sh
python3 -m streamlit run agora_topic_modeling/webapp/webapp.py
```

