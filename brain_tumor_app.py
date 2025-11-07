import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Simple app without complex dependencies
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI image for tumor type prediction")

# Sidebar with better spacing
st.sidebar.title("About")
st.sidebar.info(
    "This AI tool classifies brain MRI images into 4 categories:\n\n"
    "• **Glioma**\n\n"
    "• **Meningioma**\n\n" 
    "• **No Tumor**\n\n"
    "• **Pituitary**\n\n"
    "Upload a brain MRI image for instant classification."
)

# Add more space in sidebar
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.subheader("Model Info")
st.sidebar.write("Deep Learning Model trained on 2,443 MRI scans")
st.sidebar.write("**Accuracy:** >90% on test data")

# Class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('best_brain_tumor_model.h5')
    except:
        st.error("Model not found! Please check if 'best_brain_tumor_model.h5' exists.")
        return None

model = load_model()

# File upload
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    if st.button("Classify Image"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])  # Convert to Python float
            
        st.success(f"**Prediction:** {class_names[predicted_class]}")
        st.success(f"**Confidence:** {confidence:.1%}")  # Fixed percentage formatting
        
        # Show all probabilities with progress bars
        st.write("**Detailed Confidence Scores:**")
        
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            prob_float = float(prob)  # Convert to Python float
            
            # Create columns for label and progress bar
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write(f"{class_name}:")
            
            with col2:
                # Color the predicted class differently
                color = "#FF4B4B" if i == predicted_class else "#1F77B4"
                
                # Display progress bar
                st.progress(prob_float)
                
                # Display percentage text
                st.write(f"{prob_float:.1%}")

else:
    st.info("Please upload a brain MRI image to get started.")

st.markdown("---")
st.caption("Note: For educational purposes only. Consult healthcare professionals for medical diagnoses.")