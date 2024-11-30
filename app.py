import streamlit as st
import google.generativeai as ai
import torch
import numpy as np
from PIL import Image
import easyocr
from gtts import gTTS
import os

# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="centered")
st.title("Visual Assistance AI 👓🤖")
st.header("Empowering Visually Impaired Individuals 🧠⚡")

# Initialize Generative AI
try:
    api_key = st.secrets["AI_API_KEY"]
    ai.configure(api_key=api_key)
    sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Provide detailed descriptions for visually impaired individuals."""
    model_scene = ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_prompt_scene)
except KeyError:
    st.error("AI_API_KEY is missing in Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Generative AI: {e}")
    st.stop()

# Initialize YOLOv5 and EasyOCR
@st.cache_resource
def load_resources():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        reader = easyocr.Reader(['en'])
        return model, reader
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, reader = load_resources()

# File uploader
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Feature selection
features = {
    "scene_understanding": "Real-Time Scene Understanding",
    "text_to_speech": "Text-to-Speech Conversion",
    "object_detection": "Object and Obstacle Detection"
}
selected_features = [key for key, value in features.items() if st.checkbox(value)]

# Process uploaded image
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Scene Understanding
    if "scene_understanding" in selected_features:
        st.subheader("Scene Understanding")
        try:
            description_prompt = "Describe this scene in detail for a visually impaired individual."
            response = model_scene.generate_content(description_prompt)
            st.write("**Scene Description:**", response.text if hasattr(response, 'text') else "Failed to generate description.")
        except Exception as e:
            st.error(f"Error during scene understanding: {e}")

    # Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            result = reader.readtext(np.array(image))
            extracted_text = " ".join([text[1] for text in result])
            if extracted_text.strip() == "":
                st.warning("No text found in the image.")
            else:
                st.write("**Extracted Text:**", extracted_text)
                tts = gTTS(text=extracted_text, lang='en')
                tts.save("output_audio.mp3")
                st.audio("output_audio.mp3")
        except Exception as e:
            st.error(f"Error during text-to-speech conversion: {e}")

    # Object Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            results = model(np.array(image))
            detected_objects = [results.names[int(label)] for label in results.xyxy[0][:, -1].tolist()]
            description = "I see the following objects in the image: " + ", ".join(detected_objects)
            st.write(description)
        except Exception as e:
            st.error(f"Error during object detection: {e}")
