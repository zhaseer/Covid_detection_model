import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# --- CONFIGURATION ---
# Define the three possible outcomes
CLASS_NAMES = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}
# The image size the model expects (224x224 pixels)
IMAGE_SIZE = 224


# --- MODEL LOADING ---
# Use cache so the model only loads once
@st.cache_resource
def load_tuned_model():
    # Load the Keras model file. 
    # We include custom_objects to correctly load the VGG16 base.
    return tf.keras.models.load_model(
        "tuned_ai_model_best_lat.keras",
        custom_objects={'VGG16': tf.keras.applications.VGG16}
    )

# --- PREDICTION LOGIC ---
def run_prediction(image_file, model):
    """Processes the image and gets the diagnosis from the model."""
    try:
        # 1. Load and prepare the image
        image = Image.open(image_file).convert("RGB")
        img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        # Add a dimension for the batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Normalize pixel values (0 to 1)
        img_array = img_array / 255.0
        
        # 2. Make prediction
        # The result is an array of probabilities for all three classes
        prediction_probabilities = model.predict(img_array).flatten()
        
        # 3. Find the most likely class
        class_index = np.argmax(prediction_probabilities)
        predicted_name = CLASS_NAMES[class_index]
        predicted_prob = prediction_probabilities[class_index]
        
        return predicted_name, predicted_prob
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Return None if any error happens
        return None, None

# --- STREAMLIT INTERFACE ---

st.title("COVID Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image for diagnosis (Normal, Viral Pneumonia, or COVID).")

# Attempt to load the model and stop if it fails
try:
    model = load_tuned_model()
except Exception as e:
    st.error("Model Loading Failed. Please check dependencies and model file.")
    st.stop()

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
    
    # Run prediction when the button is clicked
    if st.button("Predict Diagnosis", type="primary"):
        
        # Run the prediction logic
        predicted_name, predicted_prob = run_prediction(uploaded_file, model)
        
        if predicted_name:
            st.markdown("---")
            st.subheader("Predicted Diagnosis")
            
            # Display the result simply (no emojis)
            if predicted_name == 'Covid':
                st.error(f"Result: **{predicted_name}**")
            else:
                st.success(f"Result: **{predicted_name}**")
            
            # Use the expander to show probability on click
            with st.expander(f"View Confidence Score for {predicted_name}"):
                st.markdown(f"Confidence: **{predicted_prob*100:.2f}%**")