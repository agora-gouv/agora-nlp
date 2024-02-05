## Agora-NLP

Open response analysis project set up by state-owned start-up Agora.


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

```bash 
# Create a new environment named as you would like 
python3 -m venv new_env
```

```bash 
# Load the virtual environment you created previously
source new_env/bin/activate
```

```bash
# Install the required dependencies that have been stored in a file for the occasion
pip install -r install/requirements.txt
```

### Variables d'environnements

- TOPIC_THRESHOLD_FOR_SUBTOPIC: Valeur en pourcentage utilisés pour savoir à partir de quel représentation total des données par un topic est-ce qu'on calcul ses sous-topics. Exemple si TOPIC_THRESHOLD_FOR_SUBTOPIC=5 alors on calcul les sous-topics pour tous les topics représentant au moins 5% des données de réponses.


### Déploiement
Utilisation de la plateforme Scalingo pour déployer cet outil de gestion de pipeline.
Déploiement manuel de l'application sur l'interface de Scalingo en sélectionnant la branche à déployer.
Scalingo utilise un *Procfile* situé à la racine du projet pour savoir quoi lancer au démarrage de l'application déployée.
Contenu du *Procfile*:
```
web: airflow webserver
worker: airflow scheduler
```
La première commande sert 

### Usage
Se rendre 

> Author: Theo Santos