
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_path = "market_research_cluster_model.pkl"
kmeans = joblib.load(model_path)

st.title("Market Research Cluster Creation Tool")

st.write("Bu ilova mijozlarni segmentlash uchun KMeans modelidan foydalanadi.")

# Input form for user data
with st.form(key="user_input"):
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    income = st.number_input("Annual Income (k$)", min_value=0, step=1)
    spending_score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, step=1)
    submit = st.form_submit_button("Predict Cluster")

# If form is submitted
if submit:
    user_data = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending_score]
    })
    cluster = kmeans.predict(user_data)[0]
    st.success(f"Ushbu mijoz {cluster} klasterga tegishli!")

st.write("Model bilan ishlash uchun .pkl fayli yuklangan bo'lishi kerak.")
