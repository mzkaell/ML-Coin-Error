import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("coin_model.keras")

model = load_model()

# Predict function
def predict_coin(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    return ("Double Die", confidence) if confidence > 0.5 else ("Single Die", 1 - confidence)

# Streamlit app layout
st.set_page_config(page_title="CoinSight - Error Detector", layout="centered")
st.title("🪙 CoinSight - Coin Error Detection")
st.write("Upload a coin image and CoinSight will predict if it's a **Double Die** or **Single Die**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_coin(image_pil)
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")
