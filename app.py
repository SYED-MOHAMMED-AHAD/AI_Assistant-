import streamlit as st
import google.generativeai as ai
from PIL import Image
import pytesseract
from gtts import gTTS
import numpy as np
import torch
import os
import io
import pyttsx3
import easyocr
reader = easyocr.Reader(['en'])  # Load English language model

import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR"
elif platform.system() == "Linux":
    # On Linux servers, Tesseract is usually installed at this path
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
else:
    raise EnvironmentError("Unsupported platform for Tesseract OCR")


# Configure API key for Google Generative AI from environment variables (for security)
ai.configure(api_key='AIzaSyA88kc5SzNdPLdTlsdIN2xs8CBz_HLMdy8')  

import streamlit as st

# Set the page layout to wide for more space
st.set_page_config(page_title="Visual Assistance AI 👓🤖", layout="wide")

# Title of the app
st.title("AI-Powered Visual Assistance for the Visually Impaired 👓🤖")

# Create two columns
col1, col2 = st.columns([1, 3])  # 1 for left section, 3 for right section

# Left Column - About Me and Project
with col1:
    st.header("About Me 👤")
    st.write("""
    I am a passionate developer and AI enthusiast, working on projects that combine technology 
    and accessibility. This project is designed to empower visually impaired individuals using 
    cutting-edge AI tools, such as object detection, scene understanding, and text-to-speech 
    conversion, to enhance their daily lives.
    """)

    st.header("About This Project 💡")
    st.write("""
    The Visual Assistance AI project aims to provide visually impaired individuals with real-time 
    information about their surroundings. Using object detection, the system identifies objects in 
    the environment and provides a description via text or speech. 
    Features include:
    - **Real-Time Scene Understanding**
    - **Object and Obstacle Detection**
    - **Text-to-Speech Conversion**
    
    By leveraging AI technologies like YOLOv5 for object detection and Google Generative AI for scene 
    understanding, the project aims to bridge the gap in accessibility and provide meaningful support.
    """)

# Right Column - Main App Features
with col2:
    st.header("Visual Assistance AI in Action 🧠⚡")
    st.write("""
    Upload an image to get real-time assistance. The AI model will detect objects, describe the scene, 
    and even convert any text in the image to speech.
    """)

    # File uploader for the image
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    # Feature selection checkboxes
    st.subheader("Select Features:")
    features = {
        "scene_understanding": "Real-Time Scene Understanding",
        "text_to_speech": "Text-to-Speech Conversion",
        "object_detection": "Object and Obstacle Detection"
    }
    
    selected_features = []
    for key, value in features.items():
        if st.checkbox(value):
            selected_features.append(key)

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Add your object detection and other features logic here
        # For instance, perform object detection, scene understanding, or text-to-speech conversion



# Streamlit app setup
st.set_page_config(page_title="Visual Assistance AI 🤖", layout="centered")
st.title("Visual Assistance AI 🤖")
st.header("Intelligent Vision Assistance for All 🌐")

# Initialize the object detection model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model for object detection

# Define the system prompt for scene understanding
sys_prompt_scene = """You are an advanced AI specializing in scene understanding. Based on the content of the provided text, 
describe the scene in detail to help visually impaired individuals comprehend their surroundings."""

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

            # Create a description from the detected objects
            description = "I see the following objects in the image: " + ", ".join(detected_objects)

            # Generate scene description using Generative AI (using detected objects)
            description_prompt = f"Describe the following scene for a visually impaired person: {description}"
            response = model_scene.generate_content(description_prompt)
            
            if response and hasattr(response, 'text') and response.text:
                st.write("Scene Description:", response.text)
            else:
                st.error("Failed to generate scene description.")
        except Exception as e:
            st.error(f"An error occurred during scene understanding: {e}")



    # Perform Text-to-Speech Conversion
    if "text_to_speech" in selected_features:
        st.subheader("Text-to-Speech Conversion")
        try:
            # Extract text from the image using EasyOCR
            result = reader.readtext(np.array(image))  # Use EasyOCR to process the image
            extracted_text = " ".join([text[1] for text in result])  # Combine extracted text
            
            if extracted_text.strip() == "":  # Handle case where no text is extracted
                st.warning("No text found in the image.")
            else:
                st.write("Extracted Text:", extracted_text)
                # Convert extracted text to speech using Google Text-to-Speech
                tts = gTTS(text=extracted_text, lang='en')
                # Save audio output
                audio_path = "output_audio.mp3"
                tts.save(audio_path)  # Save the audio file
                st.audio(audio_path)  # Play the audio file in Streamlit
        except Exception as e:
            st.error(f"Error during text-to-speech conversion: {e}")




    # Perform Object and Obstacle Detection
    if "object_detection" in selected_features:
        st.subheader("Object and Obstacle Detection")
        try:
            # Perform object detection using YOLOv5 model
            results = model(np.array(image))
            annotated_image = np.array(results.render()[0])  # Annotate image with bounding boxes
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during object detection: {e}")
