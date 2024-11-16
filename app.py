import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans

# Modelni yuklash
@st.cache_resource
def load_model():
    with open('cluster_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sarlavha
st.title("Clustering Model App")
st.write("Ushbu ilova yuklangan ma'lumotlarni klasterlash uchun foydalaniladi.")

# CSV faylni yuklash
uploaded_file = st.file_uploader("CSV faylni yuklang", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Yuklangan Ma'lumotlar")
    st.write(data.head())

    # Kiritilgan ustunlarni tekshirish
    st.write("Eslatma: Model kiritilgan ma'lumot ustunlari o'qitilgan ma'lumotlar ustunlariga mos kelishi kerak.")

    # Klasterlashni bajarish
    if st.button("Run Clustering"):
        try:
            # NaN qiymatlarni tozalash
            data_cleaned = data.dropna()
            
            # Klasterlash
            predictions = model.predict(data_cleaned)
            data_cleaned['Cluster'] = predictions
            
            st.subheader("Klasterlangan Ma'lumotlar")
            st.write(data_cleaned)
            
            # CSV formatda yuklash uchun tugma
            csv = data_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Data as CSV",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Xatolik yuz berdi: {e}")
else:
    st.info("Iltimos, CSV fayl yuklang.")
