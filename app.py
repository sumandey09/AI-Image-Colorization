import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AI Colorization",
    layout="wide"
)

st.title("🎨 AI Image & Video Colorization")

@st.cache_resource
def load_ai_model():
    return load_model(
        "colorization_model.h5",
        compile=False
    )

model = load_ai_model()

feature = st.sidebar.selectbox(
    "Choose Feature",
    [
        "Image Colorization",
        "Video Colorization"
    ]
)

# ==========================
# IMAGE COLORIZATION
# ==========================

if feature == "Image Colorization":

    st.subheader("🖼️ Image Colorization")

    uploaded_file = st.file_uploader(
        "Upload a Black & White Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")
        image_np = np.array(image)

        resized = cv2.resize(
            image_np,
            (128, 128)
        )

        normalized = resized.astype(
            "float32"
        ) / 255.0

        model_input = normalized.reshape(
            1,
            128,
            128,
            1
        )

        prediction = model.predict(
            model_input,
            verbose=0
        )

        output = prediction[0]

        output = np.clip(
            output,
            0,
            1
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(
                "Input B&W Image"
            )
            st.image(
                image_np,
                use_container_width=True
            )

        with col2:
            st.subheader(
                "Colorized Output"
            )
            st.image(
                output,
                use_container_width=True
            )

        st.success(
            "Image Colorization Completed!"
        )

# ==========================
# VIDEO COLORIZATION
# ==========================

if feature == "Video Colorization":

    st.subheader("🎥 Video Colorization")

    uploaded_video = st.file_uploader(
        "Upload a Black & White Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(
            uploaded_video.read()
        )

        video_path = temp_video.name

        st.info(
            "Processing Video... Please Wait ⏳"
        )

        cap = cv2.VideoCapture(
            video_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        cap.release()

        colorized_frames = []

        progress_bar = st.progress(0)

        for idx, frame in enumerate(frames):

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                (128, 128)
            )

            gray = gray.astype(
                "float32"
            ) / 255.0

            gray = gray.reshape(
                1,
                128,
                128,
                1
            )

            pred = model.predict(
                gray,
                verbose=0
            )

            output = pred[0]

            output = np.clip(
                output,
                0,
                1
            )

            output = (
                output * 255
            ).astype(np.uint8)

            output = cv2.resize(
                output,
                (width, height)
            )

            colorized_frames.append(
                output
            )

            progress_bar.progress(
                (idx + 1) / len(frames)
            )

        output_video = "colorized_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(
            *'mp4v'
        )

        out = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        for frame in colorized_frames:

            out.write(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )
            )

        out.release()

        st.success(
            "Video Colorization Completed!"
        )

        st.video(
            output_video
        )

        with open(
            output_video,
            "rb"
        ) as file:

            st.download_button(
                label="⬇ Download Colorized Video",
                data=file,
                file_name="colorized_video.mp4",
                mime="video/mp4"
            )