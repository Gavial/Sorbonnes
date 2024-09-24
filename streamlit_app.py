import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests


# Define the FastAPI endpoint URL
API_URL = 'http://127.0.0.1:8000/predict'

# Configuration de la page
st.set_page_config(page_title="Analyse des Données de Santé Cardiaque", layout="wide")

# Centrer le titre
st.markdown("<h1 style='text-align: center;'>Analyse des Données de Santé Cardiaque</h1>", unsafe_allow_html=True)


# Chargement des données
df = pd.read_csv('heart_disease_uci.csv')


# Liste des pages
pages = ["Descriptif","Dictionnaire","Prédiction", "Caractéristiques", "Importance des caractéristiques", "Corrélation", "Distribution", "Répartition"]

# Sélection de la page
selected_page = st.sidebar.radio("Choisissez une page :", pages)


# Page "Descriptif"
if selected_page == "Descriptif":
    st.header("Description du projet")
    st.subheader("Prédiction et Visualisation des Maladies Cardiaques")
    st.write("""Développer un tableau de bord interactif pour prédire les risques de maladies cardiaques en fonction des données de santé des patients.""")
    st.markdown("""1. **Données Kaggle** :
    - `Heart Disease UCI` : [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)""")
    st.markdown("""2. **Technologies** :
    - `Python` : Jupyter Notebook
    - `Streamlit` : pour le tableau de bord interactif
    - `Scikit-Learn` : pour les modèles de machine learning
    - `Matplotlib/Seaborn` : pour les visualisations""")
    st.markdown("""3. **Fonctionnalités** :
    - `Visualisation des données` : Distribution des caractéristiques des patients (âge, sexe, pression artérielle, etc.).
    - `Modèles de prédiction` : Utilisation de modèles de machine learning (par exemple, logistic regression, decision tree, random forest) pour prédire les risques de maladies cardiaques.
    - `Interface utilisateur` : Permettre aux utilisateurs de saisir leurs données de santé et d'obtenir une prédiction du risque.
    - `Analyse des caractéristiques` : Afficher les caractéristiques les plus influentes pour les prédictions.""")
    st.markdown("""4. **Avantages** :
    - `Pertinence` : Correspond à notre apprentissage en machine learning, deep learning, et traitement de données.
    - `Impact` : Utilité pratique et impact potentiel sur la santé publique.
    - `Accessibilité` : Données disponibles gratuitement sur Kaggle, outils utilisés gratuits.""")


# Page "Dictionnaire"
if selected_page == "Dictionnaire":
    st.header("Dictionnaire des Colonnes du Dataset")

    st.write("""
    Description des colonnes présentes dans le dataset `heart_disease_uci.csv` :
    """)

    st.write("1. **id** : Identifiant unique pour chaque patient. Cette colonne est utilisée pour l'identification et n'a pas de valeur dans les analyses statistiques.")

    st.write("2. **age** : Âge du patient en années. Cette caractéristique est importante car l'âge est un facteur de risque significatif pour les maladies cardiaques.")

    st.write("3. **sex** : Sexe du patient. C’est un facteur de risque pour les maladies cardiaques, avec des différences dans la prévalence et la gravité entre les sexes.")

    st.write("4. **dataset** : Lieu d'études.")

    st.markdown("""5. **cp** : Type de douleur thoracique. Il s'agit d'une variable catégorielle avec plusieurs niveaux :
    - Douleur angineuse typique
    - Douleur angineuse atypique
    - Douleur non angineuse
    - Pas de douleur
    """)

    st.write("6. **trestbps** : Pression artérielle au repos en mm Hg.")

    st.write("7. **chol** : Taux de cholestérol sérique en mg/dl. Un taux élevé de cholestérol est associé à un risque accru de maladies cardiaques.")

    st.markdown("""8. **fbs** : Glycémie à jeun (fbs > 120 mg/dl) :
    - `True` : Glycémie élevée
    - `False` : Glycémie normale
    """)

    st.markdown("""9. **restecg** : Résultats de l'électrocardiogramme au repos :
    - Normal
    - Présence de signes d'une anomalie st
    - Présence de signes d'une anomalie hypertrophie ventriculaire gauche
    """)

    st.write("10. **thalch** : Fréquence cardiaque maximale atteinte pendant l'exercice.")

    st.markdown("""11. **exang** : Indication d'angine induite par l'exercice :
    - `True` : Oui
    - `False` : Non
    """)

    st.write("12. **oldpeak** : Dépression du segment ST induite par l'exercice par rapport au repos. Cela peut indiquer une maladie cardiaque si la dépression est significative.")

    st.markdown("""13. **slope** : Pente du segment ST au pic de l'exercice :
    - Pente ascendante
    - Pente plate
    - Pente descendante
    """)

    st.write("14. **ca** : Nombre de vaisseaux principaux (0-3) colorés par fluoroscopie. Plus le nombre est élevé, plus il y a de vaisseaux obstructifs.")

    st.markdown("""15. **thal** : Thalassémie (anomalie génétique) :
    - Normal
    - Fixation défectueuse
    - Fixation réduite
    """)

    st.markdown("""16. **num** : Niveau de la présence de la maladie cardiaque :
    - `0` : Absence de maladie cardiaque
    - `1` : Présence de maladie cardiaque faible
    - `2` : Présence de maladie cardiaque moyenne
    - `3` : Présence de maladie cardiaque grave
    - `4` : Présence de maladie cardiaque sévère
    """)


# Page "Prédiction"
if selected_page == "Prédiction":
    st.header("Entrez vos données de santé ci-dessous pour obtenir une prédiction :")

    # Création du formulaire de saisie
    id = st.number_input("ID", min_value=0)
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", [0, 1])  # Example: 0 for female, 1 for male
    dataset = st.text_input("Dataset", value="Hungary")
    cp = st.selectbox("CP", [1, 2, 3, 4])  # Example values
    trestbps = st.number_input("Trestbps", min_value=0.0)
    chol = st.number_input("Chol", min_value=0.0)
    fbs = st.checkbox("Fbs")
    restecg = st.selectbox("Restecg", ["normal", "abnormal"])  # Example values
    thalch = st.number_input("Thalch", min_value=0.0)
    exang = st.checkbox("Exang")
    oldpeak = st.number_input("Oldpeak", min_value=0.0)

    # Bouton de soumission
    if st.button('Soumettre'):
        # Prepare the payload
        payload = {
            'id': id,
            'age': age,
            'sex': sex,
            'dataset': dataset,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak
        }
        
        # Send the request to the API
        response = requests.get(API_URL, params=payload)
        
        # Display the prediction
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error:", response.status_code, response.text)

