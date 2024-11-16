import streamlit as st
import pickle
import numpy as np

# Modelni yuklash
with open('cluster_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interfeysi
st.title("KMeans Cluster Model Deployment")
st.write("Modelga ma'lumot kiritib, klasterni aniqlang:")

# Feature kiritish maydonlari
inputs = []
for i in range(4):  # 4 ta feature uchun
    value = st.number_input(f"Feature {i+1}", step=0.01, format="%.2f")
    inputs.append(value)

# Modelga kirish
if st.button("Predict Cluster"):
    try:
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        st.success(f"Model cluster bashorati: {prediction[0]}")
    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")
