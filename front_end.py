import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
st.set_page_config(page_title="Skin Analyzer", page_icon="🧴", layout="wide")
st.markdown("""
<style>
            .stApp{
            background-color:#cc66ff;
            }
            h1{
            text-align:center;
            color:#003300;
            }
            .result-box{
            padding:15px;
            border-radius:10px;
            background-color:#e8f5e9;
            font-size:20px;
            text-align:center;
            }
            </style>
            """,unsafe_allow_html=True)

st.title("AI Skin Condition Classifier")
uploaded_file=st.file_uploader("Upload Face Image", type=["jpg","jpeg","png"])

# Load model with Keras 3.x compatibility
# The .keras format is the modern Keras format compatible with Keras 3.x
@st.cache_resource
def load_model_cached():
    """Load and cache the model to avoid reloading on every interaction"""
    try:
        # Primary: Load the .keras format model (Keras 3.x compatible)
        model = tf.keras.models.load_model(
            "module4_results/efficientnet_face_skin_model_final.keras",
            compile=False,
            safe_mode=False
        )
        st.success("✓ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load .keras model: {str(e)[:200]}")
        #Fallback: Try loading the .h5 format
        try:
            model = tf.keras.models.load_model(
                "module3(p2)_results/efficientnet_skin_model.h5",
                compile=False,
                safe_mode=False
            )
            st.warning("⚠️ Loaded fallback .h5 model")
            return model
        except Exception as e2:
            st.error(f"Failed to load all models: {str(e2)[:200]}")
            raise

# Load the model
model = load_model_cached()
classes=["clear_skin","dark_spots","puffy_eyes", "wrinkles"]

def predict_skin(image):
    image=image.resize((224,224))      
    img=np.array(image)                
    img=preprocess_input(img)          
    img=np.expand_dims(img, axis=0)   
    prediction = model.predict(img)
    class_index=np.argmax(prediction)
    return classes[class_index]

if uploaded_file is not None:
    image=Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)
    st.header("Analyzing skin...")
    result = predict_skin(image)
    st.markdown(
        f'<div class="result-box">Prediction: <b>{result}</b></div>',
        unsafe_allow_html=True
    )