import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import PIL

# Load model
model = load_model("tb_detector_model.h5")

# Class labels
class_names = ['Normal', 'Tuberculosis']

# Streamlit UI
st.title("🩻 Tuberculosis Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict if it shows signs of Tuberculosis.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display image
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded X-ray', use_container_width=True)

    # Preprocess
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]
    threshold = 0.4  # try 0.45 or 0.4
    result = class_names[int(prediction > threshold)]
    st.markdown(f"Model confidence: `{prediction:.4f}`")
    st.markdown(f"### 🧠 Prediction: **{result}**")
    st.markdown(f"Model confidence: `{prediction:.4f}`")


