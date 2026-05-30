import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Image Colorization", layout="wide")

st.title("🎨 Black & White Image Colorization")
st.write("Upload a grayscale image and convert it into a color image using AI.")

@st.cache_resource
def load_ai_model():
    return load_model("colorization_model.h5", compile=False)

model = load_ai_model()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (128, 128))
    normalized = resized.astype("float32") / 255.0

    model_input = normalized.reshape(1, 128, 128, 1)

    prediction = model.predict(model_input)

    output = prediction[0]
    output = np.clip(output, 0, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input B&W Image")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Colorized Output")
        st.image(output, use_container_width=True)

    st.success("Colorization Completed!")