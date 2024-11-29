import streamlit as st
import google.generativeai as ai
from PIL import Image
import pytesseract
from gtts import gTTS
import numpy as np
import torch

# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="centered")
st.title("Visual Assistance AI 👓🤖")
st.header("Empowering Visually Impaired Individuals 🧠⚡")

# Fetch API key from Streamlit Secrets Manager
try:
    api_key = st.secrets["AI_API_KEY"]
    ai.configure(api_key=api_key)
except KeyError:
    st.error("AI_API_KEY is missing in Streamlit secrets. Please add it.")
    st.stop()

# Initialize the object detection model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model
except Exception as e:
    st.error(f"Error loading YOLOv5 model: {e}")
    st.stop()

# Define the system prompt for scene understanding
sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Based on an image's content, provide a detailed and accurate description that helps visually impaired individuals comprehend their surroundings."""

try:
    model_scene = ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_prompt_scene)
except Exception as e:
    st.error(f"Error initializing Generative AI model: {e}")
    st.stop()

# Define feature options
features = {
    "scene_understanding": "Real-Time Scene Understanding",
    "text_to_speech": "Text-to-Speech Conversion",
    "object_detection": "Object and Obstacle Detection"
}

# File uploader for the image
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Feature selection
st.subheader("Select Features:")
selected_features = [key for key, value in features.items() if st.checkbox(value)]

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if "scene_understanding" in selected_features:
        st.subheader("Scene Understanding")
        try:
            description_prompt = "Describe this scene in detail for a visually impaired individual."
            response = model_scene.generate_content(description_prompt)
            if response and hasattr(response, 'text') and response.text:
                st.write("**Scene Description:**", response.text)
            else:
                st.error("Failed to generate scene description.")
        except Exception as e:
            st.error(f"Scene understanding error: {e}")

    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                st.write("**Extracted Text:**", extracted_text)
                tts = gTTS(text=extracted_text, lang='en')
                tts.save("output_audio.mp3")
                st.audio("output_audio.mp3")
            else:
                st.warning("No text found in the image.")
        except Exception as e:
            st.error(f"An error occurred during text-to-speech conversion: {e}")

    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            results = model(np.array(image))
            if len(results.xywh) > 0:
                detected_objects = [results.names[int(label)] for label in results.xywh[0][:, -1].tolist()]
                description = "I see the following objects in the image: " + ", ".join(detected_objects)
                st.write(description)
            else:
                st.warning("No objects detected in the image.")
        except Exception as e:
            st.error(f"An error occurred during object detection: {e}")
