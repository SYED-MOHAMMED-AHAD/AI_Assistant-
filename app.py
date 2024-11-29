import streamlit as st
import google.generativeai as ai
from PIL import Image
import easyocr  # Using EasyOCR instead of pytesseract
from gtts import gTTS
import numpy as np
import torch
import os
import pyttsx3  # pyttsx3 added for offline TTS

# Configure API key for Google Generative AI from Streamlit Secrets
try:
    api_key = st.secrets["AI_API_KEY"]
    ai.configure(api_key=api_key)
except KeyError:
    st.error("AI_API_KEY is missing in Streamlit secrets. Please add it.")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="centered")
st.title("Visual Assistance AI 👓🤖")
st.header("Empowering Visually Impaired Individuals 🧠⚡")

# Initialize the object detection model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model for object detection

# Define the system prompt for scene understanding
sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Based on the content of the provided text, describe the scene in detail to help visually impaired individuals comprehend their surroundings."""

# Initialize the Generative AI model for scene understanding
model_scene = ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_prompt_scene)

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

# Initialize EasyOCR reader (for OCR text extraction)
reader = easyocr.Reader(['en'])

# Function to display an image with a caption
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
            # Perform object detection using YOLOv5 model
            results = model(np.array(image))
            
            # Map the detected labels to actual class names
            detected_objects = [results.names[int(label)] for label in results.xywh[0][:, -1].tolist()]
            confidence_scores = results.xywh[0][:, 4].tolist()  # Extract confidence scores (probabilities)

            # Create a description from the detected objects
            description = "I see the following objects in the image: "
            for obj, score in zip(detected_objects, confidence_scores):
                description += f"{obj} ({score*100:.2f}% confidence), "
            description = description.rstrip(", ")

            # Generate scene description using Generative AI (using detected objects)
            description_prompt = f"Describe the following scene for a visually impaired person: {description}"
            response = model_scene.generate_content(description_prompt)
            
            if response and hasattr(response, 'text') and response.text:
                st.write("*Scene Description:*", response.text)
            else:
                st.error("Failed to generate scene description.")
        except Exception as e:
            st.error(f"An error occurred during scene understanding: {e}")

    # Perform Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            # Extract text from the image using EasyOCR
            result = reader.readtext(np.array(image))
            extracted_text = " ".join([text[1] for text in result])
            
            if extracted_text.strip() == "":  # Handle case if OCR does not extract any text
                st.warning("No text found in the image.")
            else:
                st.write("*Extracted Text:*", extracted_text)
                # Convert text to speech using Google Text-to-Speech
                tts = gTTS(text=extracted_text, lang='en')
                # Save audio in a safe directory (e.g., working directory)
                audio_path = "output_audio.mp3"
                tts.save(audio_path)  # Save audio as mp3
                st.audio(audio_path)  # Play the audio
        except Exception as e:
            st.error(f"An error occurred during text-to-speech conversion: {e}")

    # Perform Object and Obstacle Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            # Perform object detection using YOLOv5 model
            results = model(np.array(image))
            annotated_image = np.array(results.render()[0])  # Annotate image with bounding boxes
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
            
            # Show the list of detected objects and confidence scores
            detected_objects = [results.names[int(label)] for label in results.xywh[0][:, -1].tolist()]
            confidence_scores = results.xywh[0][:, 4].tolist()
            description = "I see the following objects in the image: "
            for obj, score in zip(detected_objects, confidence_scores):
                description += f"{obj} ({score*100:.2f}% confidence), "
            description = description.rstrip(", ")
            st.write(description)
            
        except Exception as e:
            st.error(f"An error occurred during object detection: {e}")
