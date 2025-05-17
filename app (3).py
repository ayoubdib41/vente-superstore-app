
import streamlit as st
import pandas as pd
import joblib

st.title("📊 Application de prédiction des ventes - Superstore")

model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

categories = ['Furniture', 'Office Supplies', 'Technology']
sub_categories = ['Chairs', 'Phones', 'Binders', 'Paper', 'Storage', 'Accessories']
regions = ['East', 'West', 'South', 'Central']
segments = ['Consumer', 'Corporate', 'Home Office']
ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

st.subheader("🧾 Entrez les informations pour prédire la vente :")

year = st.number_input("Année de commande", min_value=2015, max_value=2025, value=2018)
month = st.selectbox("Mois de commande", list(range(1, 13)))
quantity = st.slider("Quantité commandée", 1, 100, 1)
discount_pct = st.slider("Remise (%)", 0, 100, 10)
discount = discount_pct / 100
profit = st.number_input("Profit estimé", value=100.0)

category = st.selectbox("Catégorie", categories)
subcategory = st.selectbox("Sous-catégorie", sub_categories)
region = st.selectbox("Région", regions)
segment = st.selectbox("Segment", segments)
ship_mode = st.selectbox("Mode de livraison", ship_modes)

if st.button("🤖 Prédire les ventes"):
    X = pd.DataFrame([[year, month, quantity, discount, profit]],
                     columns=["Order_Year", "Order_Month", "Quantity", "Discount", "Profit"])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"📈 Vente prédite : {prediction[0]:.2f} $")
