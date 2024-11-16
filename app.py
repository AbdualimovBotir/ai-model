import streamlit as st
import pickle

# Modelni yuklash
@st.cache_resource
def load_model():
    with open("cluster_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit interfeysi
st.title("Model Deployment with Streamlit")

# Foydalanuvchi ma'lumotlarini kiritish
st.write("Enter the features for prediction:")

# Input maydonlarini yaratish
num_features = st.number_input("Number of features", min_value=1, value=3)
features = []
for i in range(num_features):
    feature_value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(feature_value)

if st.button("Predict"):
    try:
        prediction = model.predict([features])
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
