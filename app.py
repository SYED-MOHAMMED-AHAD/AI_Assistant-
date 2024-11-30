import streamlit as st
import torch
import numpy as np
from PIL import Image
import easyocr  # For OCR
from gtts import gTTS  # For text-to-speech
import cv2
import tempfile

# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="centered")
st.title("Visual Assistance AI 👓🤖")
st.header("Empowering Visually Impaired Individuals 🧠⚡")

# Load YOLOv5 model for object detection
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.stop()

model = load_model()

# Initialize EasyOCR reader for text extraction
@st.cache_resource
def init_easyocr():
    return easyocr.Reader(['en'])

reader = init_easyocr()

# File uploader
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Feature selection
st.subheader("Select Features:")
features = {
    "scene_understanding": "Real-Time Scene Understanding",
    "text_to_speech": "Text-to-Speech Conversion",
    "object_detection": "Object and Obstacle Detection"
}
selected_features = [key for key, value in features.items() if st.checkbox(value)]

# Process the uploaded image
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image)

    # Real-Time Scene Understanding
    if "scene_understanding" in selected_features:
        st.subheader("Scene Understanding")
        try:
            # Define a placeholder for scene description
            st.info("Scene understanding is under development. Replace this section with Generative AI integration.")
        except Exception as e:
            st.error(f"Error during scene understanding: {e}")

    # Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            # Extract text using EasyOCR
            result = reader.readtext(image_np)
            extracted_text = " ".join([text[1] for text in result])
            if extracted_text.strip():
                st.write("**Extracted Text:**", extracted_text)
                # Convert text to speech
                tts = gTTS(text=extracted_text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    tts.save(temp_audio.name)
                    st.audio(temp_audio.name, format="audio/mp3")
            else:
                st.warning("No text found in the image.")
        except Exception as e:
            st.error(f"Error during text-to-speech conversion: {e}")

    # Object and Obstacle Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            # Perform object detection
            results = model(image_np)
            annotated_image = results.render()[0]  # Get image with bounding boxes
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Display annotated image
            st.image(annotated_image, caption="Detected Objects with Bounding Boxes", use_column_width=True)

            # Extract detection details
            detection_results = results.pandas().xyxy[0]
            if detection_results.empty:
                st.warning("No objects detected in the image.")
            else:
                st.write("**Detected Objects:**")
                for index, row in detection_results.iterrows():
                    obj_name = row["name"]
                    confidence = round(row["confidence"] * 100, 2)  # Scale confidence to 100
                    st.write(f"- **{obj_name.capitalize()}**: {confidence}%")
        except Exception as e:
            st.error(f"Error during object detection: {e}")
else:
    st.info("Please upload an image to proceed.")

