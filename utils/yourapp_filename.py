import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import tensorflow as tf
import librosa
import torch
from torchvision import transforms
from convnext_image import ConvNeXt

# ----------------------------
# SETUP & CUSTOM STYLING
# ----------------------------

st.set_page_config(page_title="AVIA - Deepfake Detection", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3, h4 {
        color: #58a6ff;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        background: rgba(56, 61, 74, 0.8);
        border: 1px solid #30363d;
        margin-top: 20px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #238636);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# SESSION STATE INITIALIZATION
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {}  # In-memory store: {username: password}

# ----------------------------
# AUTHENTICATION FUNCTIONS
# ----------------------------

def login_form():
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            users = st.session_state.users
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

def signup_form():
    st.subheader("Sign Up")
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Choose a username", key="signup_username")
        password = st.text_input("Choose a password", type="password", key="signup_password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            users = st.session_state.users
            if username in users:
                st.error("Username already exists. Please choose another.")
            else:
                users[username] = password
                st.success("Sign up successful! Please login.")

def logout():
    st.session_state.logged_in = False
    st.success("You have logged out.")

# ----------------------------
# AUDIO MODEL & HELPERS
# ----------------------------

# Custom BatchNormalization patch
class FixedBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        axis = kwargs.get("axis")
        if isinstance(axis, list) and len(axis) > 0:
            kwargs["axis"] = axis[0]
        super(FixedBatchNormalization, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        axis = config.get("axis")
        if isinstance(axis, list) and len(axis) > 0:
            config["axis"] = axis[0]
        return cls(**config)

    def get_config(self):
        config = super(FixedBatchNormalization, self).get_config()
        axis = config.get("axis")
        if isinstance(axis, list) and len(axis) > 0:
            config["axis"] = axis[0]
        return config

tf.keras.utils.get_custom_objects()['BatchNormalization'] = FixedBatchNormalization

@st.cache_resource
def load_audio_model():
    try:
        model = tf.keras.models.load_model(
            "updated_model.keras",  # update with your model path (.keras or .h5)
            custom_objects={"BatchNormalization": FixedBatchNormalization},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading audio model: {e}")
        return None

def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# ----------------------------
# IMAGE MODEL & HELPERS
# ----------------------------

def ConvNeXt_model():
    model_conv = ConvNeXt()
    state_dict = torch.load('convnext_tiny_1k_224_ema.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_conv.load_state_dict(state_dict["model"])
    return model_conv

def transform_single_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def test_single_image(model, image, device, temperature=2.0):
    model.eval()
    image_tensor = transform_single_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output / temperature, dim=1).cpu().numpy()
        predicted_class = output.argmax(dim=1).item()
        confidence_score = probabilities[0][predicted_class] * 100

    class_labels = {0: "Fake", 1: "Real"}
    predicted_label = class_labels.get(predicted_class, "Unknown")
    return predicted_label, confidence_score

# ----------------------------
# MAIN APPLICATION
# ----------------------------

def detection_interface():
    st.title("🤖 AVIA: Deepfake Detection")
    st.markdown("Detect deepfakes in audio and images using advanced AI-powered analysis.")
    st.sidebar.title("About AVIA")
    st.sidebar.info(
        "AVIA uses state-of-the-art deep learning models to analyze media and detect deepfakes. Choose a media type below to begin your analysis."
    )
    st.sidebar.title("How It Works")
    st.sidebar.markdown("""
    1. Select the detection type (Audio or Image).  
    2. Upload your media file.  
    3. Click the detection button.  
    4. Review the prediction and confidence score.
    """)

    # Detection mode switch
    mode = st.radio("Select Detection Type", ["Audio Detection", "Image Detection"], index=0)

    # ---------- Audio Detection ----------
    if mode == "Audio Detection":
        st.header("🎙️ Audio Deepfake Detection")
        audio_file = st.file_uploader("Upload Audio (supported: wav, mp3, ogg)", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file, format="audio/wav")
            if st.button("Analyze Audio"):
                audio_model = load_audio_model()
                if audio_model:
                    features = extract_features_from_audio(audio_file.getvalue())
                    if features is not None:
                        prediction = audio_model.predict(features)
                        confidence = prediction[0][0]
                        if confidence > 0.5:
                            result = f"<p style='color:#ff7b72; font-size:1.2rem;'>Deepfake Audio Detected (Confidence: {confidence*100:.2f}%)</p>"
                        else:
                            result = f"<p style='color:#3fb950; font-size:1.2rem;'>Real Audio (Confidence: {(1-confidence)*100:.2f}%)</p>"
                        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                        st.progress(confidence)

    # ---------- Image Detection ----------
    elif mode == "Image Detection":
        st.header("🖼️ Image Deepfake Detection")
        image_file = st.file_uploader("Upload Image (supported: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if image_file:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Analyze Image"):
                checkpoint_path = 'checkpoint_epoch_20 (2).pth'
                if not os.path.exists(checkpoint_path):
                    st.error(f"Checkpoint not found at {checkpoint_path}")
                else:
                    try:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = ConvNeXt_model()
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        if 'model_state_dict' not in checkpoint:
                            st.error("Invalid checkpoint format.")
                        else:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model = model.to(device)
                            predicted_label, confidence_score = test_single_image(model, image, device)
                            if predicted_label.lower() == "fake":
                                result = f"<p style='color:#ff7b72; font-size:1.2rem;'>Deepfake Image Detected (Confidence: {confidence_score:.2f}%)</p>"
                            else:
                                result = f"<p style='color:#3fb950; font-size:1.2rem;'>Real Image (Confidence: {confidence_score:.2f}%)</p>"
                            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during model inference: {e}")

# ----------------------------
# APP NAVIGATION LOGIC
# ----------------------------

def main():
    st.sidebar.title("User Authentication")
    
    if not st.session_state.logged_in:
        auth_mode = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])
        if auth_mode == "Login":
            login_form()
        else:
            signup_form()
    else:
        st.sidebar.write(f"Logged in as: **{list(st.session_state.users.keys())[0]}**")
        if st.sidebar.button("Logout"):
            logout()
    
    # Only show detection interface when logged in
    if st.session_state.logged_in:
        detection_interface()

if __name__ == "__main__":
    main()
