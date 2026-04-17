import streamlit as st
import pickle
import pandas as pd

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_model():
    with open("Model (6).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------ Page Config ------------------ #
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ------------------ Header ------------------ #
st.title("🤖 Smart Prediction App")
st.markdown("### Enter input values to get predictions")

st.markdown("---")

# ------------------ Dynamic Input Section ------------------ #
st.subheader("📥 Input Features")

# 👉 CHANGE THESE BASED ON YOUR MODEL
# Example generic inputs (you MUST edit names if needed)
feature_names = ["Feature1", "Feature2", "Feature3"]

input_data = {}

cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# ------------------ Prediction ------------------ #
st.markdown("---")

if st.button("🚀 Predict", use_container_width=True):
    try:
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)

        st.success(f"✅ Prediction Result: **{prediction[0]}**")

    except Exception as e:
        st.error("❌ Something went wrong!")
        st.exception(e)

# ------------------ Footer ------------------ #
st.markdown("---")
st.caption("Built with Streamlit • Clean UI • Fast Deployment")
