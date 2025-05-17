import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🛍️ Prédiction des Ventes - Superstore", layout="centered")

st.title("📊 Application de prédiction des ventes - Superstore")

# Chargement du modèle et du scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("🧾 Entrez les informations pour prédire la vente :")

# Champs de saisie
year = st.number_input("Année de commande", min_value=2015, max_value=2025, value=2018)
month = st.selectbox("Mois de commande", list(range(1, 13)))
quantity = st.slider("Quantité commandée", 1, 100, 1)
discount = st.slider("Remise (entre 0.0 et 1.0)", 0.0, 1.0, step=0.05)
profit = st.number_input("Profit estimé", value=50.0)

# Champs texte catégoriels
category = st.selectbox("Catégorie", ["Furniture", "Office Supplies", "Technology"])
subcategory = st.text_input("Sous-catégorie")
region = st.selectbox("Région", ["West", "East", "Central", "South"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
customer_id = st.text_input("ID du client")
ship_mode = st.selectbox("Mode de livraison", ["First Class", "Second Class", "Standard Class", "Same Day"])

# Lancer la prédiction
if st.button("🔮 Prédire les ventes"):
    input_data = pd.DataFrame([[year, month, quantity, discount, profit]],
                              columns=["Order_Year", "Order_Month", "Quantity", "Discount", "Profit"])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"📈 Vente prédite : {prediction[0]:.2f} $")
