import streamlit as st
import google.generativeai as ai
from PIL import Image
import pytesseract
from gtts import gTTS
import numpy as np
import torch
import pyttsx3
import io

# Tesseract path setup for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR"

# Configure API key for Google Generative AI
ai.configure(api_key='AIzaSyA88kc5SzNdPLdTlsdIN2xs8CBz_HLMdy8')

# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="centered")
st.title("Visual Assistance AI 👓🤖")
st.header("Empowering Visually Impaired Individuals 🧠⚡")

# Load YOLOv5 model for object detection
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = load_model()

# Initialize pyttsx3 for text-to-speech
tts_engine = pyttsx3.init()

# Define the system prompt for scene understanding
sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Based on the content of the provided text, describe the scene in detail to help visually impaired individuals comprehend their surroundings."""

# Feature options
features = {
    "scene_understanding": "Real-Time Scene Understanding",
    "text_to_speech": "Text-to-Speech Conversion",
    "object_detection": "Object and Obstacle Detection"
}

# File uploader for the image
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Feature selection
st.subheader("Select Features:")
selected_features = []
for key, value in features.items():
    if st.checkbox(value):
        selected_features.append(key)

# Process the uploaded image
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform Real-Time Scene Understanding
    if "scene_understanding" in selected_features:
        st.subheader("Scene Understanding")
        try:
            # Perform object detection
            results = model(np.array(image))
            detected_objects = [results.names[int(label)] for label in results.xywh[0][:, -1].tolist()]
            description = "I see the following objects in the image: " + ", ".join(detected_objects)

            # Use Generative AI to create a detailed scene description
            model_scene = ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_prompt_scene)
            response = model_scene.generate_content(f"Describe the scene: {description}")
            scene_description = response.text if response and hasattr(response, 'text') else "No description available."
            st.write("**Scene Description:**", scene_description)
        except Exception as e:
            st.error(f"An error occurred during scene understanding: {e}")

    # Perform Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            # Extract text from the image using OCR
            extracted_text = pytesseract.image_to_string(image).strip()
            if not extracted_text:
                st.warning("No text found in the image.")
            else:
                st.write("**Extracted Text:**", extracted_text)

                # Read the text aloud using pyttsx3
                tts_engine.say(extracted_text)
                tts_engine.runAndWait()
                st.success("Text read aloud successfully!")
        except Exception as e:
            st.error(f"An error occurred during text-to-speech conversion: {e}")

    # Perform Object and Obstacle Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            # Perform object detection
            results = model(np.array(image))
            annotated_image = np.array(results.render()[0])
            st.image(annotated_image, caption="Detected Objects with Bounding Boxes", use_column_width=True)

            # Extract detected objects and confidence scores
            detection_results = results.pandas().xyxy[0]
            if detection_results.empty:
                st.warning("No objects detected in the image.")
            else:
                st.write("**Detected Objects:**")
                for _, row in detection_results.iterrows():
                    obj_name = row["name"].capitalize()
                    confidence = round(row["confidence"] * 100, 2)
                    st.write(f"- **{obj_name}**: {confidence}%")
        except Exception as e:
            st.error(f"An error occurred during object detection: {e}")
