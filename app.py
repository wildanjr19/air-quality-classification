import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Set page config
st.set_page_config(
    page_title="AirPolNet",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained CNN model from 'model' folder."""
    model_dir = 'model/VGG16'
    keras_path = os.path.join(model_dir, 'model_VGG16.keras')
    h5_path = os.path.join(model_dir, 'model_VGG16.h5')

    if os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists(h5_path):
        model_path = h5_path
    else:
        st.error("Model file not found! Please ensure 'model_VGG16.keras' or 'model_VGG16.h5' is in the 'model' folder.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Define class labels based on your dataset structure
CLASS_LABELS = {
    0: "Baik",
    1: "Sedang", 
    2: "Tidak Sehat",
    3: "Sangat Tidak Sehat"
}

# Define colors for each class
CLASS_COLORS = {
    "Baik": "#4CAF50",  # Green
    "Sedang": "#FFC107",  # Yellow
    "Tidak Sehat": "#FF9800",  # Orange
    "Sangat Tidak Sehat": "#F44336"  # Red
}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to 224x224
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_air_quality(model, image):
    """Make prediction on the preprocessed image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get all class probabilities
        class_probabilities = {CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))}
        
        return predicted_class, confidence, class_probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    # Title and description
    st.title("🌤️ Prediksi Kualitas Udara Berdasar Gambar")
    st.markdown("Unggah gambar langit di sekitar anda untuk memprediksi tingkat kualitas udara.")
    
    # Sidebar
    st.sidebar.header("Informasi Model")
    st.sidebar.info("""
    **AirPolNet:**
    - Arsitektur: CNN (Convolutional Neural Network)
    - Framework: TensorFlow/Keras
    - Ukuran Input: 224x224 piksel
    - Kelas: 4 kategori kualitas udara
    """)

    st.sidebar.header("Kategori Kualitas Udara")
    for label, color in CLASS_COLORS.items():
        st.sidebar.markdown(f'<div style="background-color: {color}; padding: 5px; margin: 2px; border-radius: 3px; color: white; text-align: center;">{label}</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to predict its air quality level"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Display image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, class_probabilities = predict_air_quality(model, image)
            
            if predicted_class is not None:
                # Display main prediction
                predicted_label = CLASS_LABELS[predicted_class]
                color = CLASS_COLORS[predicted_label]
                
                st.markdown(f'<div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">'
                           f'<h2 style="color: white; margin: 0;">🎯 {predicted_label}</h2>'
                           f'<p style="color: white; margin: 5px 0;">Confidence: {confidence:.2%}</p>'
                           f'</div>', unsafe_allow_html=True)
                
                # Display all class probabilities
                st.subheader("📊 All Class Probabilities")
                
                # Sort probabilities in descending order
                sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
                
                for label, prob in sorted_probs:
                    color = CLASS_COLORS[label]
                    # Create progress bar with custom color
                    st.write(f"**{label}**")
                    st.progress(prob)
                    st.write(f"{prob:.2%}")
                    st.write("")
                
                # Additional information
                st.subheader("ℹ️ Interpretation")
                if predicted_label == "Baik":
                    st.success("🟢 Kualitas udara baik. Aman untuk melakukan aktivitas di luar ruangan")
                elif predicted_label == "Sedang":
                    st.warning("🟡 Air quality is moderate. Sensitive individuals should consider limiting outdoor activities.")
                elif predicted_label == "Tidak Sehat":
                    st.warning("🟠 Air quality is unhealthy. Everyone should limit outdoor activities.")
                else:  # Sangat Tidak Sehat
                    st.error("🔴 Air quality is very unhealthy. Avoid outdoor activities and stay indoors.")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Upload an Image**: Click 'Browse files' and select an image from your device
    2. **Wait for Processing**: The model will analyze the image automatically
    3. **View Results**: See the predicted air quality category and confidence score
    4. **Interpret**: Use the color-coded results and recommendations provided
    
    **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: gray;">'
        'Air Quality Prediction Model v4 | Built with Streamlit & TensorFlow'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()