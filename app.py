#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import io
from transformers import DetrForObjectDetection, DetrImageProcessor

# Check if Firebase app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase using the credentials JSON file
    cred = credentials.Certificate("/Users/pradeepramani/Desktop/firebase_proj/iot-project-99f97-firebase-adminsdk-wkvmx-a428ce6a45.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'iot-project-99f97.appspot.com'})

# Function to upload image to Firebase Storage
def upload_image_to_storage(image_bytes, storage_path):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        blob.upload_from_string(image_bytes, content_type='image/jpeg')
        return blob.public_url
    except Exception as e:
        # Log the error
        st.error(f"Error uploading image to Firebase Storage: {str(e)}")
        return None

# Load the DETR object detection model and its associated image processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

def predict_persons(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    num_persons = sum(1 for label in outputs.logits.argmax(-1).squeeze() if label.item() == 1)  # Label 1 corresponds to "person"
    return num_persons

# Define the Streamlit app
def main():
    st.title("Person Detection App")
    st.markdown("### To Detect and count persons in images")
    st.markdown("#### (Developed by: Diya Ramani)")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Read uploaded image bytes
            image_bytes = uploaded_file.read()
            
            # Display uploaded image
            st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
            
            # Button to process image
            if st.button("Detect Persons"):
                try:
                    # Make prediction
                    num_persons = predict_persons(image_bytes)
                    
                    # Print debugging statements
                    st.write("Number of persons detected:", num_persons)
                
                    # Upload image to Firebase Storage
                    image_url = upload_image_to_storage(image_bytes, "images/" + uploaded_file.name)
                    
                    if image_url:
                        # Display result
                        st.success(f"Number of persons detected: {num_persons}")
                        st.info(f"Image uploaded to Firebase Storage: {image_url}")
                    else:
                        st.error("Failed to upload image.")
                except Exception as e:
                    # Print error message
                    st.error(f"Error during prediction: {str(e)}")
        except Exception as e:
            # Print error message
            st.error(f"Error reading the uploaded image: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

