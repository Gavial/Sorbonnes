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

# Page "Prédiction"
if selected_page == "Prédiction":
    st.header("Entrez vos données de santé ci-dessous pour obtenir une prédiction :")

    # Création du formulaire de saisie avec des valeurs optionnelles (None si vide)
    id = st.number_input("ID", min_value=0, value=0)
    age = st.number_input("Age", min_value=0, value=0)
    sex = st.selectbox("Sex", [None, 0, 1], format_func=lambda x: "Sélectionnez" if x is None else "Femme" if x == 0 else "Homme")
    dataset = st.text_input("Dataset", value="Hungary")
    cp = st.selectbox("CP", [None, 1, 2, 3, 4], format_func=lambda x: "Sélectionnez" if x is None else str(x))  # CP est optionnel
    trestbps = st.number_input("Trestbps", min_value=0.0, value=0.0)
    chol = st.number_input("Chol", min_value=0.0, value=0.0)
    fbs = st.checkbox("Fbs")
    restecg = st.selectbox("Restecg", [None, "normal", "abnormal"], format_func=lambda x: "Sélectionnez" if x is None else x)
    thalch = st.number_input("Thalch", min_value=0.0, value=0.0)
    exang = st.checkbox("Exang")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, value=0.0)

    # Bouton de soumission
    if st.button('Soumettre'):
        # Préparation des données avec gestion des champs vides (None si non rempli)
        payload = {
            'id': id,
            'age': age if age != 0 else None,
            'sex': sex,
            'dataset': dataset if dataset != "" else None,
            'cp': cp,
            'trestbps': trestbps if trestbps != 0 else None,
            'chol': chol if chol != 0 else None,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch if thalch != 0 else None,
            'exang': exang,
            'oldpeak': oldpeak if oldpeak != 0 else None
        }
        
        # Filtrer les valeurs None du payload
        filtered_payload = {k: v for k, v in payload.items() if v is not None}
        
        # Envoi de la requête à l'API uniquement avec les données présentes
        response = requests.get(API_URL, params=filtered_payload)
        
        # Affichage de la prédiction
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            st.write(f"Prédiction : {prediction}")
        else:
            st.write("Erreur :", response.status_code, response.text)
