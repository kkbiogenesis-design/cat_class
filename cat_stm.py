
import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Image Classification",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Image Classification App")
st.write("Upload an image to predict its class.")

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("CAT.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((64, 64))
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)

    # Predict
    if st.button("Predict"):

        with st.spinner("Predicting..."):
            predictions = model.predict(x)

            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(predictions[0])) * 100

        # Display result
        st.success(f"Predicted Class: {predicted_class}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Prediction Scores")

        for i, score in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {score * 100:.2f}%")
            st.progress(float(score))
