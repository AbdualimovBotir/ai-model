import streamlit as st
import pandas as pd
import pickle

# Sarlavha
st.title("Clustering Model App")
st.write("Ushbu ilova klasterlash modelidan foydalanib prognozlarni ko'rsatadi.")

# Modelni yuklash
@st.cache_resource
def load_model():
    with open('cluster_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# CSV faylni yuklash
uploaded_file = st.file_uploader("Upload your CSV file for clustering", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(data.head())

    # Klasterlashni bajarish
    if st.button("Run Clustering"):
        try:
            predictions = model.predict(data)
            data['Cluster'] = predictions
            st.subheader("Clustered Data")
            st.write(data)
            
            # CSV yuklab olish uchun link
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Data as CSV",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Model bilan ishlashda xatolik yuz berdi: {e}")
else:
    st.info("Iltimos, CSV faylni yuklang.")
