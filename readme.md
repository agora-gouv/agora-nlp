## Agora-NLP

Open response analysis project set up by state-owned start-up Agora.


## Structure du répertoire

- **agora_topic_modeling** : dossier contenant le code l'analyse de texte
  - **code** : dossier du code des pages
  - **dags** : dossier de code utilisé par les page notamment pour récupérer des données ou faire du traitement de données avant leur affichage
  -*.ipynb : Notebook contenant du code pouvant servir à l'exécution des pipelines de manière manuelle 

- **Procfile** : fichier utile au déploiement sur Scalingo
- **requirement.txt** : fichier contenant les dépendances des librairies pythons


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

### Usage

**WIP**

> Author: Theo Santos