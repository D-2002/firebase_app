#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import io
from transformers import DetrForObjectDetection, DetrImageProcessor

# Initialize Firebase using the credentials JSON file
cred = credentials.Certificate("/Users/pradeepramani/Desktop/firebase_proj/iot-project-99f97-firebase-adminsdk-wkvmx-59268366fb.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'iot-project-99f97.appspot.com'})

# Load the DETR object detection model and its associated image processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Function to predict number of persons in an image
def predict_persons(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    num_persons = sum(1 for label in outputs.logits.argmax(-1).squeeze() if label.item() == 1)  # Label 0 corresponds to "person"
    return num_persons

# Define the Streamlit app
def main():
    st.title("Person Detection App")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read uploaded image bytes
        image_bytes = uploaded_file.read()
        
        # Display uploaded image
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        
        # Button to process image
        if st.button("Detect Persons"):
            # Make prediction
            num_persons = predict_persons(image_bytes)
            
            # Display result
            st.success(f"Number of persons detected: {num_persons}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
