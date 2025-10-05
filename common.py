import streamlit as st
import datetime, io, os, tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
import librosa
import torch
import torch.nn as nn
from torchvision import transforms

from model.pred_func import load_genconvit, df_face, pred_vid, is_video, real_or_fake
from model.config import load_config

# ──────────────────────────────────────────────────────────────────────────────
# UI & Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AVIA - Deepfake Detection", page_icon="🤖", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
body {font-family:'Roboto',sans-serif;background:#0d1117;color:#c9d1d9;}
h1,h2,h3,h4 {color:#58a6ff;}
.stButton>button{background:#238636;color:#fff;border:0;border-radius:5px;padding:10px 20px;font-size:1.05rem;transition:.3s;}
.stButton>button:hover{background:#2ea043;}
[data-testid="stSidebar"]{background:#161b22;}
.result-box{border-radius:10px;padding:20px;background:rgba(56,61,74,.85);border:1px solid #30363d;margin-top:20px;text-align:center;}
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#58a6ff,#238636);}
</style>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Firebase Setup
# ──────────────────────────────────────────────────────────────────────────────
import pyrebase
from firebase_config import firebase_config
from firebase_admin_connect import db

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
for k in ("user_email", "idToken", "uid"):
    st.session_state.setdefault(k, None)

# ──────────────────────────────────────────────────────────────────────────────
# Audio Model Helpers
# ──────────────────────────────────────────────────────────────────────────────
class FixedBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self,*a,**kw):
        kw["axis"] = kw.get("axis", -1) if isinstance(kw.get("axis"), list) else kw.get("axis")
        super().__init__(*a, **kw)
tf.keras.utils.get_custom_objects()['BatchNormalization'] = FixedBatchNormalization

@st.cache_resource
def load_audio_model():
    return tf.keras.models.load_model(
        "my_model.h5",
        custom_objects={"BatchNormalization": FixedBatchNormalization},
        compile=False
    )

def extract_features(audio_bytes, sr=16000, n_mfcc=40, max_len=500):
    y,_ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.reshape(1,*mfcc.shape,1)

# ──────────────────────────────────────────────────────────────────────────────
# Image Models
# ──────────────────────────────────────────────────────────────────────────────
def ConvNeXt_model():
    from convnext_image import ConvNeXt
    model_conv = ConvNeXt()
    state_dict = torch.load('convnext_tiny_1k_224_ema.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_conv.load_state_dict(state_dict["model"])
    return model_conv

def ConvNeXt_model_tech():
    model = ConvNeXt_model()
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 5)
    model.head = model.head.to('cpu')
    return model

def transform_single_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def test_tech_single_image(model, image, device, temperature=2.0):
    model.eval()
    image_tensor = transform_single_image(image).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output / temperature, dim=1).cpu().numpy()
        predicted_class = output.argmax(dim=1).item()
        confidence_score = probabilities[0][predicted_class] * 100
    tech_labels = {
        0: "denoising_diffusion_gan",
        1: "stable_diffusion",
        2: "star_gan",
        3: "stylegan2",
        4: "pro_gan"
    }
    predicted_tech = tech_labels.get(predicted_class, "Unknown")
    return predicted_tech, confidence_score

# ──────────────────────────────────────────────────────────────────────────────
# Video Model Helpers
# ──────────────────────────────────────────────────────────────────────────────
def predict_video_file(video_path, model, num_frames=15, net="genconvit", fp16=False):
    try:
        if is_video(video_path):
            df = df_face(video_path, num_frames, net)
            if fp16:
                df = df.half()
            if len(df) >= 1:
                y, conf = pred_vid(df, model)
            else:
                y, conf = 0, 0.0
            return y, conf
        else:
            st.error("Uploaded file is not a valid video.")
            return None, None
    except Exception as e:
        st.error(f"An error occurred while processing the video: {str(e)}")
        return None, None

def remove_key_recursively(data, key_to_remove):
    if isinstance(data, dict):
        data.pop(key_to_remove, None)
        for k, v in data.items():
            remove_key_recursively(v, key_to_remove)
# ----------------------------
# MAIN APPLICATION INTERFACE
# ----------------------------
def detection_interface():
    st.title("🤖 AVIA: Deepfake Detection")
    st.markdown("Detect deepfakes in audio, images, and videos using advanced AI-powered analysis.")
    st.sidebar.title("About AVIA")
    st.sidebar.info(
        "AVIA uses state-of-the-art deep learning models to analyze media and detect deepfakes. Choose a media type below to begin your analysis."
    )
    st.sidebar.title("How It Works")
    st.sidebar.markdown("""
    1. Select the detection type (Audio, Image or Video).  
    2. Upload your media file.  
    3. Click the detection button.  
    4. Review the prediction and confidence score.
    """)
    
    mode = st.radio("Select Detection Type", ["Audio Detection", "Image Detection", "Video Detection"], index=0)
    
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
                        st.progress(float(confidence))
    
    elif mode == "Image Detection":
        st.header("🖼️ Image Deepfake Detection")
        image_file = st.file_uploader("Upload Image (supported: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if image_file:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Image"):
                # Step 1: Real vs Fake prediction
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
                                st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                                # Step 2: Technology Classification
                                tech_checkpoint = 'convnext2_epoch_20.pth'
                                if not os.path.exists(tech_checkpoint):
                                    st.error(f"Technology checkpoint not found at {tech_checkpoint}")
                                else:
                                    model2 = ConvNeXt_model_tech()
                                    checkpoint2 = torch.load(tech_checkpoint, map_location=device)
                                    if 'model_state_dict' not in checkpoint2:
                                        st.error("Invalid technology checkpoint format.")
                                    else:
                                        model2.load_state_dict(checkpoint2['model_state_dict'])
                                        model2 = model2.to(device)
                                        predicted_tech, tech_conf = test_tech_single_image(model2, image, device)
                                        st.write(f"**Deepfake Technology:** {predicted_tech}")
                                        st.write(f"**Technology Confidence:** {tech_conf:.2f}%")
                            else:
                                result = f"<p style='color:#3fb950; font-size:1.2rem;'>Real Image (Confidence: {confidence_score:.2f}%)</p>"
                                st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during model inference: {e}")
    
    elif mode == "Video Detection":
        video_detection()

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
        st.sidebar.write(f"Logged in as: *{list(st.session_state.users.keys())[0]}*")
        if st.sidebar.button("Logout"):
            logout()
    
    if st.session_state.logged_in:
        detection_interface()

if __name__ == "__main__":
    main()
