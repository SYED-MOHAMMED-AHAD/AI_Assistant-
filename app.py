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
api_key = st.secrets["AI_API_KEY"]

# Configure API key for Google Generative AI
ai.configure(api_key=api_key)  

# Initialize the object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model for object detection

# Define the system prompt for scene understanding
sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Based on an image's content, provide a detailed and accurate description that helps visually impaired individuals comprehend their surroundings."""

# Initialize the Generative AI model for scene understanding
model_scene = ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_prompt_scene)

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
selected_features = []
for key, value in features.items():
    if st.checkbox(value):
        selected_features.append(key)

# Function to display an image with caption
def display_image_with_caption(image, caption):
    st.image(image, caption=caption, use_column_width=True)

# Process the uploaded image
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Perform Real-Time Scene Understanding
    if "scene_understanding" in selected_features:
        st.subheader("Scene Understanding")
        try:
            # Generate scene description using Generative AI
            description_prompt = "Describe this scene in detail for a visually impaired individual."
            response = model_scene.generate_content(description_prompt)
            if response and hasattr(response, 'text') and response.text:
                st.write("**Scene Description:**", response.text)
            else:
                st.error("Failed to generate scene description.")
        except Exception as e:
            st.error(f"An error occurred during scene understanding: {e}")

    # Perform Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            # Extract text from the image using OCR
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip() == "":  # Handle case if OCR does not extract any text
                st.warning("No text found in the image.")
            else:
                st.write("**Extracted Text:**", extracted_text)
                # Convert text to speech using Google Text-to-Speech
                tts = gTTS(text=extracted_text, lang='en')
                tts.save("output_audio.mp3")  # Save audio as mp3
                st.audio("output_audio.mp3")  
        except Exception as e:
            st.error(f"An error occurred during text-to-speech conversion: {e}")

    # Perform Object and Obstacle Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            # Perform object detection using YOLOv5 model
            results = model(np.array(image))

            # Map the detected labels to actual class names
            detected_objects = [results.names[int(label)] for label in results.xywh[0][:, -1].tolist()]

            # Create a description from the detected objects
            description = "I see the following objects in the image: " + ", ".join(detected_objects)
            st.write(description)
        except Exception as e:
            st.error(f"An error occurred during object detection: {e}")
