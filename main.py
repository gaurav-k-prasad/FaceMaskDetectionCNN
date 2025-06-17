import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("mask_model.keras")  # Make sure this file is in the same directory

# Set your model's expected input size (change if needed)
IMG_SIZE = (32, 32)

# Streamlit UI
st.title("😷 Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image.resize((256, 256)), caption="Uploaded Image")

    # Preprocess the image
    img_resized = image.resize(IMG_SIZE)
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]  # assuming output is sigmoid (single neuron)

    # Display result
    if prediction >= 0.5:
        st.error("❌ Person is **not wearing a mask**.")
    else:
        st.success("✅ Person is **wearing a mask**.")
