import streamlit as st
from PIL import Image
import io
import os
import datetime
import numpy as np
import tensorflow as tf
import librosa
import torch
from torchvision import transforms
import tempfile
import torch.nn as nn

# Firebase imports
import pyrebase
from firebase_config import firebase_config
from firebase_admin_connect import db

# Video detection imports
from model.pred_func import load_genconvit, df_face, pred_vid, is_video, real_or_fake
from model.config import load_config

# ---------------------------------------------------------------------------- #
# 1) PAGE CONFIG & STYLING
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
# 2) FIREBASE INITIALIZATION
# ---------------------------------------------------------------------------- #
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# ---------------------------------------------------------------------------- #
# 3) SESSION STATE FOR AUTH
# ---------------------------------------------------------------------------- #
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "idToken" not in st.session_state:
    st.session_state.idToken = None
if "uid" not in st.session_state:
    st.session_state.uid = None

# ---------------------------------------------------------------------------- #
# 4) AUTHENTICATION HANDLERS
# ---------------------------------------------------------------------------- #
def handle_signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.user_email = email
        st.session_state.idToken    = user["idToken"]
        st.session_state.uid        = user["localId"]
        st.success("✅ Signup successful!")
    except Exception as e:
        st.error(f"❌ Signup failed: {e}")

def handle_login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user_email = email
        st.session_state.idToken    = user["idToken"]
        st.session_state.uid        = user["localId"]
        st.success("✅ Login successful!")
    except Exception as e:
        st.error(f"❌ Login failed: {e}")

def handle_logout():
    st.session_state.user_email = None
    st.session_state.idToken    = None
    st.session_state.uid        = None
    st.success("🔒 Logged out")

# ---------------------------------------------------------------------------- #
# 5) FIRESTORE WRITE HELPER
# ---------------------------------------------------------------------------- #
def store_result(media_type, filename, prediction, confidence):
    """Saves a detection record to Firestore if the user is logged in."""
    if st.session_state.user_email:
        data = {
            "uid":        st.session_state.uid,
            "user":       st.session_state.user_email,
            "filename":   filename,
            "type":       media_type,
            "result":     prediction,
            "confidence": float(confidence),
            "timestamp":  datetime.datetime.utcnow().isoformat()
        }
        try:
            db.collection("detections").add(data)
            st.success("💾 Result saved to Firestore")
        except Exception as e:
            st.error(f"⚠️ Could not save to Firestore: {e}")

# ---------------------------------------------------------------------------- #
# 6) MODEL & PREPROCESSING HELPERS (unchanged)
# ---------------------------------------------------------------------------- #
class FixedBatchNormalization(tf.keras.layers.BatchNormalization):
    def _init_(self, *args, **kwargs):
        axis = kwargs.get("axis")
        if isinstance(axis, list) and len(axis) > 0:
            kwargs["axis"] = axis[0]
        super(FixedBatchNormalization, self)._init_(*args, **kwargs)

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
        return tf.keras.models.load_model(
            "my_model.h5",
            custom_objects={"BatchNormalization": FixedBatchNormalization},
            compile=False
        )
    except Exception as e:
        st.error(f"Error loading audio model: {e}")
        return None

def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0,0),(0,max_length-mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:,:max_length]
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def ConvNeXt_model():
    from convnext_image import ConvNeXt
    model_conv = ConvNeXt()
    state = torch.load('convnext_tiny_1k_224_ema.pth',
                       map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_conv.load_state_dict(state["model"])
    return model_conv

def transform_single_image(image):
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    return t(image).unsqueeze(0)

def test_single_image(model, image, device, temperature=2.0):
    model.eval()
    img_t = transform_single_image(image).to(device)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out/temperature, dim=1).cpu().numpy()
        cls = out.argmax(dim=1).item()
        conf = probs[0][cls] * 100
    label_map = {0: "Fake", 1: "Real"}
    return label_map.get(cls,"Unknown"), conf

def test_tech_single_image(model, image, device, temperature=2.0):
    model.eval()
    img_t = transform_single_image(image).to(device)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out/temperature, dim=1).cpu().numpy()
        cls = out.argmax(dim=1).item()
        conf = probs[0][cls] * 100
    tech_map = {
        0:"denoising_diffusion_gan",1:"stable_diffusion",
        2:"star_gan",3:"stylegan2",4:"pro_gan"
    }
    return tech_map.get(cls,"Unknown"), conf

def predict_video_file(video_path, model, num_frames=15, net="genconvit", fp16=False):
    try:
        if is_video(video_path):
            df = df_face(video_path, num_frames, net)
            if fp16:
                df = df.half()
            if len(df)>=1:
                y, conf = pred_vid(df, model)
            else:
                y, conf = 0, 0.0
            return y, conf
        else:
            st.error("Not a valid video.")
            return None, None
    except Exception as e:
        st.error(f"Video error: {e}")
        return None, None

def remove_key_recursively(d, key):
    if isinstance(d, dict):
        d.pop(key, None)
        for v in d.values():
            remove_key_recursively(v, key)

# ---------------------------------------------------------------------------- #
# 7) MAIN DETECTION INTERFACE
# ---------------------------------------------------------------------------- #
def detection_interface():
    st.title("🤖 AVIA: Deepfake Detection")
    st.markdown("Choose a media type and upload to detect deepfakes.")

    mode = st.radio("Select Detection Type", ["Audio Detection", "Image Detection", "Video Detection"])

    # --- AUDIO ---
    if mode == "Audio Detection":
        audio_file = st.file_uploader("Upload Audio (wav, mp3, ogg)", type=["wav","mp3","ogg"])
        if audio_file:
            st.audio(audio_file)
            if st.button("Analyze Audio"):
                mdl = load_audio_model()
                if mdl:
                    feats = extract_features_from_audio(audio_file.getvalue())
                    if feats is not None:
                        pred = mdl.predict(feats)[0][0]
                        is_df = pred > 0.5
                        label = "Deepfake Audio Detected" if is_df else "Real Audio"
                        conf  = (pred if is_df else 1-pred) * 100
                        st.markdown(f"<div class='result-box'><p>{label} (Confidence: {conf:.2f}%)</p></div>", unsafe_allow_html=True)
                        store_result("Audio", getattr(audio_file, "name", "uploaded_audio"), label, conf)

    # --- IMAGE ---
    elif mode == "Image Detection":
        img_file = st.file_uploader("Upload Image (jpg, jpeg, png)", type=["jpg","jpeg","png"])
        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, use_column_width=True)
            if st.button("Analyze Image"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Real vs Fake
                ckpt = 'checkpoint_epoch_20 (2).pth'
                if not os.path.exists(ckpt):
                    st.error(f"Checkpoint not found: {ckpt}")
                else:
                    mdl = ConvNeXt_model().to(device)
                    cp = torch.load(ckpt, map_location=device)
                    mdl.load_state_dict(cp['model_state_dict'])
                    lbl, conf_score = test_single_image(mdl, img, device)
                    msg = f"Deepfake Image Detected" if lbl=="Fake" else "Real Image"
                    st.markdown(f"<div class='result-box'><p>{msg} (Confidence: {conf_score:.2f}%)</p></div>", unsafe_allow_html=True)
                    store_result("Image", getattr(img_file,"name","uploaded_image"), msg, conf_score)
                    # Tech classification
                    if lbl=="Fake":
                        tech_ckpt = 'convnext2_epoch_20.pth'
                        if os.path.exists(tech_ckpt):
                            mdl2 = ConvNeXt_model_tech().to(device)
                            cp2  = torch.load(tech_ckpt, map_location=device)
                            mdl2.load_state_dict(cp2['model_state_dict'])
                            tech_lbl, tech_conf = test_tech_single_image(mdl2, img, device)
                            st.write(f"**Generation Method:** {tech_lbl}  ({tech_conf:.2f}%)")
                        else:
                            st.error(f"Tech checkpoint not found: {tech_ckpt}")

    # --- VIDEO ---
    else:
        vid_file = st.file_uploader("Upload Video (avi, mp4, mpg, mpeg, mov)", type=["avi","mp4","mpg","mpeg","mov"])
        if vid_file:
            st.video(vid_file)
            num_frames = st.number_input("Frames to process", min_value=1, value=15, step=1)
            fp16       = st.checkbox("Enable FP16", value=True)
            if fp16 and not torch.cuda.is_available():
                st.warning("No GPU detected, disabling FP16.")
                fp16 = False
            if st.button("Analyze Video"):
                cfg = load_config()
                remove_key_recursively(cfg, "pretrained_cfg")
                mdl_vid = load_genconvit(cfg, net="genconvit",
                                         ed_weight="genconvit_ed_inference",
                                         vae_weight="genconvit_vae_inference",
                                         fp16=fp16)
                if fp16:
                    mdl_vid = mdl_vid.half()

                # write bytes to temp file
                vb = vid_file.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(vb)
                    tmp_path = tmp.name

                with st.spinner("Predicting..."):
                    y, c = predict_video_file(tmp_path, mdl_vid, num_frames=num_frames, net="genconvit", fp16=fp16)
                os.remove(tmp_path)

                if y is not None:
                    pred = real_or_fake(y)
                    if pred == "REAL":
                        c = 1 - c
                    conf_pct = c * 100
                    st.success(f"Prediction: {pred}")
                    st.info(f"Confidence: {conf_pct:.2f}%")
                    store_result("Video", getattr(vid_file, "name", "uploaded_video"), pred, conf_pct)
                else:
                    st.error("Prediction failed.")

# ---------------------------------------------------------------------------- #
# 8) SIDEBAR AUTH UI & APP LAUNCH
# ---------------------------------------------------------------------------- #
def main():
    st.sidebar.header("Authentication")
    if not st.session_state.user_email:
        action = st.sidebar.radio("Action", ["Login", "Sign Up"])
        email  = st.sidebar.text_input("Email", key="auth_email")
        pwd    = st.sidebar.text_input("Password", type="password", key="auth_pw")

        if action == "Login":
            if st.sidebar.button("Login"):
                handle_login(email.strip(), pwd)
        else:
            if st.sidebar.button("Sign Up"):
                handle_signup(email.strip(), pwd)
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")
        if st.sidebar.button("Logout"):
            handle_logout()

    if st.session_state.user_email:
        detection_interface()
    else:
        st.info("🔐 Please log in or sign up to use AVIA.")

if __name__ == "__main__":
    main()
