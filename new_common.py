import os, sys, types

# ── 1. Runtime guards (MUST come before any Streamlit import) ──────────────────
os.environ["TORCH_DISABLE_CUSTOM_CLASS_LOOKUP"]  = "1"      # keep your original
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"   # ← kills the crash

import torch   # ← torch *before* we inject the stub

# ── 2. Give Streamlit a harmless stub for torch.classes ────────────────────────
sys.modules["torch.classes"] = types.ModuleType("torch.classes")
sys.modules["torch.classes"].__path__ = []       # fake package path

import streamlit as st, datetime, io, tempfile, numpy as np
from PIL import Image
import tensorflow as tf, librosa, torch.nn as nn
from torchvision import transforms
from model.pred_func import load_genconvit, df_face, pred_vid, is_video, real_or_fake
from model.config import load_config
import pyrebase
from firebase_config import firebase_config
from firebase_admin_connect import db
import pandas as pd
import gdown
import json
import firebase_admin
from firebase_admin import credentials

# ─────────────────────────────  GOOGLE DRIVE MODEL DOWNLOAD ─────────────────────────────
import os
import gdown

# ──────────────────── GOOGLE DRIVE FILES ────────────────────
# logical name → Drive file ID
FILE_IDS = {
    "audio":      "1GKyDyk22eu1fuENvMRBhjXJzA0hkkdUc",
    "convnext":   "1jOYMQCD6UReZSyPYjWONbCdZLme5jrSo",
    "checkpoint": "1EXlKnNXR18WWHBQzb9txF0fub0kEcKZQ",
    "convnext2":  "1KakhvcRbMbXA46xm5u5d5b3bQuxr8EuJ",
}

# logical name → local filename
FILE_PATHS = {
    "audio":      "my_model.h5",
    "convnext":   "convnext_tiny_1k_224_ema_image.pth",
    "checkpoint": "checkpoint_epoch_20 (2).pth",
    "convnext2":  "convnext2_epoch_20.pth",
}



def _download_if_missing(name):
    out = FILE_PATHS[name]
    if not os.path.exists(out):
        url = f"https://drive.google.com/uc?id={FILE_IDS[name]}"
        gdown.download(url, out, fuzzy=True, quiet=False)

# download everything once at startup
for key in FILE_IDS:
    _download_if_missing(key)

# ------------------------------------------------------------------
# Map the helper’s FILE_PATHS to the constant names used elsewhere
# ------------------------------------------------------------------
AUDIO_MODEL_PATH  = FILE_PATHS["audio"]        #  ==> my_model.h5
CONVNEXT_PATH     = FILE_PATHS["convnext"]     #  ==> convnext_tiny_1k_224_ema_image.pth
CHECKPOINT_PATH   = FILE_PATHS["checkpoint"]   #  ==> checkpoint_epoch_20 (2).pth
CONVNEXT2_PATH    = FILE_PATHS["convnext2"]    #  ==> convnext2_epoch_20.pth

# ──────────────────────────────────────────────────────────────

# 1. Page Config (MUST be first)
st.set_page_config(page_title="AVIA - Home", layout="centered")

# ─────────────────── FIREBASE ADMIN INITIALIZATION ───────────────────
# Load service account JSON from Streamlit secrets
sa_json = json.loads(st.secrets["firebase"]["account_key"])
# Initialize Firebase Admin SDK (if not already)
if not firebase_admin._apps:
    cred = credentials.Certificate(sa_json)
    firebase_admin.initialize_app(cred)


# 2. Additional Imports
import pyrebase
from streamlit_cookies_manager import EncryptedCookieManager

# -----------------------------------------------------------------------------  
# 4. Streamlit Session State Initialization  
# -----------------------------------------------------------------------------
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "idToken" not in st.session_state:
    st.session_state.idToken = None
if "uid" not in st.session_state:
    st.session_state.uid = None
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False
if "guest_tries" not in st.session_state:
    st.session_state.guest_tries = 0

# -----------------------------------------------------------------------------  
# 5. Cookies Manager Initialization  
# -----------------------------------------------------------------------------
cookies = EncryptedCookieManager(prefix="aviadetects_", password="somestrongpassword123")
if not cookies.ready():
    st.stop()

# -----------------------------------------------------------------------------  
# 6. Pyrebase Initialization (Firebase Auth & Storage)  
# -----------------------------------------------------------------------------
firebase = pyrebase.initialize_app(firebase_config)
auth     = firebase.auth()
storage  = firebase.storage()
for k in ("user_email","idToken","uid","guest_count","show_auth","viewing_history"):
    st.session_state.setdefault(k, None)

# -----------------------------------------------------------------------------  
# 7. Load Auth Info from Cookies  
# -----------------------------------------------------------------------------
def load_cookies():
    if not st.session_state.user_email and cookies.get("user_email"):
        st.session_state.user_email = cookies.get("user_email")
    if not st.session_state.idToken and cookies.get("idToken"):
        st.session_state.idToken = cookies.get("idToken")
    if not st.session_state.uid and cookies.get("uid"):
        st.session_state.uid = cookies.get("uid")
load_cookies()

# -----------------------------------------------------------------------------  
# 8. Sidebar: Dynamic Authentication Menu  
# -----------------------------------------------------------------------------
st.sidebar.header("Authentication")

def do_logout():
    st.session_state.idToken    = None
    st.session_state.uid        = None
    st.session_state.is_guest   = False
    st.session_state.guest_tries= 0
    cookies["user_email"] = ""
    cookies["idToken"]    = ""
    cookies["uid"]        = ""
    cookies.save()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)

def handle_login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.update(
            is_guest=False,
            user_email=email,
            idToken=user["idToken"],
            uid=user["localId"],
            guest_tries=0
        )
        cookies["user_email"] = email
        cookies["idToken"]    = user["idToken"]
        cookies["uid"]        = user["localId"]
        cookies.save()
        st.sidebar.success("Login successful!")
    except Exception as e:
        st.sidebar.error(f"Login error: {e}")

def handle_signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.update(
            is_guest=False,
            user_email=email,
            idToken=user["idToken"],
            uid=user["localId"],
            guest_tries=0
        )
        cookies["user_email"] = email
        cookies["idToken"]    = user["idToken"]
        cookies["uid"]        = user["localId"]
        cookies.save()
        st.sidebar.success("Signup successful!")
    except Exception as e:
        st.sidebar.error(f"Signup error: {e}")

def handle_guest():
    st.session_state.update(user_email=None, idToken=None, uid=None, is_guest=True, guest_tries=0)
    cookies["user_email"] = ""
    cookies["idToken"]    = ""
    cookies["uid"]        = ""
    cookies.save()

if st.session_state.user_email and not st.session_state.is_guest:
    if st.sidebar.button("Logout"):
        do_logout()
elif st.session_state.is_guest:
    st.sidebar.write("Currently in Guest Mode")
    ge = st.sidebar.text_input("Email for Login").strip()
    gp = st.sidebar.text_input("Password for Login", type="password")
    if st.sidebar.button("Login as User"):
        handle_login(ge, gp)
    se = st.sidebar.text_input("Email for Signup").strip()
    sp = st.sidebar.text_input("Password for Signup", type="password")
    if st.sidebar.button("Create Account"):
        handle_signup(se, sp)
else:
    st.sidebar.write("Select an option below:")
    le = st.sidebar.text_input("Email for Login").strip()
    lp = st.sidebar.text_input("Password for Login", type="password")
    if st.sidebar.button("Login"):
        handle_login(le, lp)
    se = st.sidebar.text_input("Email for Signup").strip()
    sp = st.sidebar.text_input("Password for Signup", type="password")
    if st.sidebar.button("Signup"):
        handle_signup(se, sp)

# -----------------------------------------------------------------------------  
# 9. Sample Media Dictionary  
# -----------------------------------------------------------------------------
SAMPLE_MEDIA = {
    "Audio": [("Real Audio","samples/real_audio.wav"),("Fake Audio","samples/fake_audio.wav")],
    "Image": [("Real Image","samples/real_image.jpg"),("Fake Image","samples/fake_image.jpg")],
    "Video": [("Real Video","samples/real_video.mp4"),("Fake Video","samples/fake_video.mp4")]
}

# -----------------------------------------------------------------------------  
# 11. Guest Tries Helper  
# -----------------------------------------------------------------------------
def can_guest_detect():
    return st.session_state.guest_tries < 5
def increment_guest_tries():
    st.session_state.guest_tries += 1
def show_guest_reached_limit():
    st.warning("You have reached the maximum of 5 guest detections. Please log in or sign up for unlimited usage.")

# ───────────────────────────  THEME & STYLE  ────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600&display=swap');
    :root{--primary:#4B6FE7;--accent:#6E35E8;--canvas:#0D1A34;}
    [data-testid="stApp"]>div:first-child{max-width:85%!important;margin:0 auto!important;}
    html,body,[data-testid="stApp"]{background:radial-gradient(circle at 10% 10%,#13224A 0%,var(--canvas) 60%) fixed;}
    html,body,div,span,p,label,input,textarea{font-size:20px!important;color:#ECEFF4!important;}
    h1{font:600 7rem 'Cinzel',serif!important;letter-spacing:.07em;color:var(--primary)!important;margin-bottom:.2em;}
    button[kind="primary"],button{font-size:1.3rem!important;font-weight:600!important;color:#fff!important;
    background:linear-gradient(90deg,var(--primary),var(--accent))!important;border:none!important;border-radius:6px!important;
    box-shadow:0 2px 4px rgba(0,0,0,.15)!important;}
    button[kind="primary"]:hover,button:hover{background:linear-gradient(90deg,var(--accent),var(--primary))!important;}
    .card{background:rgba(255,255,255,.05);padding:1.5rem;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.35);}
    .stSelectbox,.stFileUploader>div{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.15)!important;
    border-radius:6px!important;}
    </style>""", unsafe_allow_html=True)

# ─────────────────────────────  HELPERS  ────────────────────────────────
class FixedBN(tf.keras.layers.BatchNormalization):
    def __init__(self,*a,**kw):
        kw["axis"] = kw.get("axis",-1) if isinstance(kw.get("axis"),list) else kw.get("axis")
        super().__init__(*a,**kw)
tf.keras.utils.get_custom_objects()['BatchNormalization'] = FixedBN

# ────────────── PATCH old-Keras "batch_shape" → new name ──────────────
# from tensorflow.keras.layers import InputLayer

# _original_from_config = InputLayer.from_config  # keep reference

# @classmethod
# def _patched_from_config(cls, config):
#     # ↪ if a legacy model uses "batch_shape", rename it
#     if "batch_shape" in config and "batch_input_shape" not in config:
#         config["batch_input_shape"] = config.pop("batch_shape")
#     return _original_from_config(config)

# # replace the method only once
# InputLayer.from_config = _patched_from_config
# ───────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_audio_model():
    return tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={"BatchNormalization":FixedBN}, compile=False)

def extract_features(b,sr=16000,n_mfcc=40,max_len=500):
    y,_ = librosa.load(io.BytesIO(b), sr=sr)
    m   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    m   = np.pad(m, ((0,0),(0,max_len-m.shape[1])), mode="constant") if m.shape[1]<max_len else m[:,:max_len]
    return m.reshape(1,*m.shape,1)

def convnext_image_model():
    from convnext_image import ConvNeXt_image
    mdl = ConvNeXt_image()
    mdl.load_state_dict(torch.load(CONVNEXT_PATH, map_location="cpu")["model"])
    return mdl

def convnext_image_model_tech():
    m = convnext_image_model()
    m.head = nn.Linear(m.head.in_features, 5)
    return m

def tf_img(img):
    t = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    return t(img).unsqueeze(0)

def test_single(model,img,dev):
    model.eval()
    with torch.no_grad():
        out = model(tf_img(img).to(dev))
        p   = torch.softmax(out,1).cpu().numpy()
    return ("Fake","Real")[out.argmax(1).item()], p[0].max()*100

def test_tech(model,img,dev):
    model.eval()
    with torch.no_grad():
        out = model(tf_img(img).to(dev))
        p   = torch.softmax(out,1).cpu().numpy()
    tech_map = {0:"denoising_diffusion_gan",1:"stable_diffusion",2:"star_gan",3:"stylegan2",4:"pro_gan"}
    return tech_map[out.argmax(1).item()], p[0].max()*100

def remove_key_recursively(d,key):
    if isinstance(d,dict):
        d.pop(key,None)
        for v in d.values(): remove_key_recursively(v,key)
    elif isinstance(d,list):
        for i in d: remove_key_recursively(i,key)

def predict_video_file(path, model, n=15, net="genconvit", fp16=False):
    if not is_video(path):
        return None, None

    df = df_face(path, n, net)              # <= might be []

 
    if isinstance(df, list) or len(df) == 0:
        return None, None                   # caller will show “Prediction failed”

    if fp16:
        df = df.half()

    return pred_vid(df, model)


def store_result(tp, filename, result, confidence, local_path=None):
    file_url = None
    if local_path:
        try:
            storage.child(f"uploads/{filename}").put(local_path, st.session_state.idToken)
            file_url = storage.child(f"uploads/{filename}").get_url(None)
            st.write("File uploaded to Storage:", file_url)
        except Exception as upload_error:
            st.error(f"Error uploading file: {upload_error}")
    if st.session_state.user_email and st.session_state.user_email != "guest":
        data = {
            "uid":        st.session_state.uid,
            "user":       st.session_state.user_email,
            "filename":   filename,
            "type":       tp,
            "result":     result,
            "confidence": float(confidence),
            "media_url":  file_url,
            "timestamp":  datetime.datetime.utcnow().isoformat()
        }
        try:
            db.collection("detections").add(data)
            st.success("Detection result stored in Firebase!")
        except Exception as firestore_error:
            st.error(f"Error storing detection data: {firestore_error}")

# ─────────────────────────────  UI PARTS  ────────────────────────────────
def auth_sidebar():
    st.sidebar.header("Authentication")
    act   = st.sidebar.radio("Action",["Login","Sign Up"])
    email = st.sidebar.text_input("Email")
    pwd   = st.sidebar.text_input("Password", type="password")
    if act=="Login" and st.sidebar.button("Login"):
        try:
            u = auth.sign_in_with_email_and_password(email, pwd)
            st.session_state.update(user_email=email, idToken=u["idToken"], uid=u["localId"], show_auth=False)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
    if act=="Sign Up" and st.sidebar.button("Sign Up"):
        try:
            u = auth.create_user_with_email_and_password(email, pwd)
            st.session_state.update(user_email=email, idToken=u["idToken"], uid=u["localId"], show_auth=False)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Signup failed: {e}")

def welcome_interface():
    st.markdown(
        "<h1 style='text-align:center'>AVIA</h1>"
        "<p style='text-align:center;margin-top:-1rem;font-size:1.3rem'>"
        "Audio-Visual Integrity Analyzer&nbsp;|&nbsp;Deepfake Detection</p><hr>",
        unsafe_allow_html=True
    )
    _, mid, _ = st.columns([1,2,1])
    with mid:
        l, r = st.columns(2)
        if l.button("🔓 Guest Mode"):
            handle_guest(); st.experimental_rerun()
        if r.button("🔐 Login / Sign Up"):
            st.session_state.show_auth = True
    if st.session_state.show_auth:
        auth_sidebar()

def detection_interface():
    st.title("AVIA")
    mode = st.selectbox("Choose Detection Type", ("Audio","Image","Video"), index=0)

    if mode=="Audio":
        uploaded = st.file_uploader("Upload Audio", type=["wav","mp3","ogg"])
    elif mode=="Image":
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    else:
        uploaded = st.file_uploader("Upload Video", type=["avi","mp4","mov","mpg","mpeg"])

    if uploaded:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if mode=="Audio":
            if st.button("Analyze Audio"):
                pred = load_audio_model().predict(extract_features(uploaded.getvalue()))[0][0]
                lbl, conf = ("Fake", pred*100) if pred>0.5 else ("Real", (1-pred)*100)
                st.success(f"{lbl} • {conf:.2f}%")
                store_result("Audio", uploaded.name, lbl, conf)

        elif mode=="Image":
            img = Image.open(uploaded).convert("RGB")
            st.image(img)
            if st.button("Analyze Image"):
                dev = torch.device("cpu")
                mdl = convnext_image_model().to(dev)
                mdl.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=dev)["model_state_dict"])
                lbl, conf = test_single(mdl, img, dev)
                st.success(f"{lbl} • {conf:.2f}%")
                store_result("Image", uploaded.name, lbl, conf)
                if lbl == "Fake":
                    mdl2 = convnext_image_model_tech().to(dev)
                    mdl2.load_state_dict(torch.load(CONVNEXT2_PATH, map_location=dev)["model_state_dict"])
                    tech, tconf = test_tech(mdl2, img, dev)
                    st.info(f"Technology: {tech} ({tconf:.2f}%)")

        else:
            st.video(uploaded)
            frames = 1
            fp16   = st.checkbox("Enable FP16", True) if torch.cuda.is_available() else False
            if st.button("Analyze Video"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded.getvalue()); tmp.close()
                cfg = load_config(); remove_key_recursively(cfg, "pretrained_cfg")
                mdl = load_genconvit(cfg, net="genconvit", ed_weight="genconvit_ed_inference",
                                     vae_weight="genconvit_vae_inference", fp16=fp16)
                if fp16: mdl = mdl.half()
                y, conf = predict_video_file(tmp.name, mdl, frames, fp16=fp16)
                os.remove(tmp.name)
                if y is not None:
                    pred = real_or_fake(y)
                    conf = 1-conf if pred=="REAL" else conf
                    st.success(f"{pred} • {conf*100:.2f}%")
                    store_result("Video", uploaded.name, pred, conf*100)
                else:
                    st.error("Prediction failed")

        st.markdown("</div>", unsafe_allow_html=True)

def detection_interface_forguest():
    st.title("AVIA")
    mode = st.selectbox("Choose Detection Type", ("Audio","Image","Video"), index=0)

    if mode=="Audio":
        uploaded = st.file_uploader("Upload Audio", type=["wav","mp3","ogg"])
    elif mode=="Image":
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    else:
        uploaded = st.file_uploader("Upload Video", type=["avi","mp4","mov","mpg","mpeg"])

    if uploaded:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if mode=="Audio":
            if st.button("Analyze Audio"):
                pred = load_audio_model().predict(extract_features(uploaded.getvalue()))[0][0]
                lbl, conf = ("Fake", pred*100) if pred>0.5 else ("Real", (1-pred)*100)
                st.success(f"{lbl} • {conf:.2f}%"); store_result("Audio", uploaded.name, lbl, conf)

        elif mode=="Image":
            img = Image.open(uploaded).convert("RGB"); st.image(img)
            if st.button("Analyze Image"):
                dev = torch.device("cpu")
                mdl = convnext_image_model().to(dev)
                mdl.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=dev)["model_state_dict"])
                lbl, conf = test_single(mdl, img, dev)
                st.success(f"{lbl} • {conf:.2f}%"); store_result("Image", uploaded.name, lbl, conf)
                if lbl == "Fake":
                    mdl2 = convnext_image_model_tech().to(dev)
                    mdl2.load_state_dict(torch.load(CONVNEXT2_PATH, map_location=dev)["model_state_dict"])
                    tech, tconf = test_tech(mdl2, img, dev)
                    st.info(f"Technology: {tech} ({tconf:.2f}%)")

        else:
            st.video(uploaded)
            frames = st.number_input("Frames to sample", 1, 60, 15)
            fp16   = st.checkbox("Enable FP16", True) if torch.cuda.is_available() else False
            if st.button("Analyze Video"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded.getvalue()); tmp.close()
                cfg = load_config(); remove_key_recursively(cfg, "pretrained_cfg")
                mdl = load_genconvit(cfg, net="genconvit", ed_weight="genconvit_ed_inference",
                                     vae_weight="genconvit_vae_inference", fp16=fp16)
                if fp16: mdl = mdl.half()
                y, conf = predict_video_file(tmp.name, mdl, frames, fp16=fp16)
                os.remove(tmp.name)
                if y is not None:
                    pred = real_or_fake(y)
                    conf = 1-conf if pred=="REAL" else conf
                    st.success(f"{pred} • {conf*100:.2f}%"); store_result("Video", uploaded.name, pred, conf*100)
                else:
                    st.error("Prediction failed")

        increment_guest_tries()
        st.info("No record is saved for guest users.")
        st.markdown("</div>", unsafe_allow_html=True)

def load_history(user_email):
    docs = db.collection("detections").where("user", "==", user_email).stream()
    history_data = []
    for d in docs:
        r = d.to_dict()
        history_data.append({
            "UID":        r.get("uid",""),
            "User":       r.get("user",""),
            "File":       r.get("filename",""),
            "Type":       r.get("type",""),
            "Result":     r.get("result",""),
            "Confidence": r.get("confidence",""),
            "Media URL":  r.get("media_url",""),
            "Timestamp":  r.get("timestamp","")
        })
    return pd.DataFrame(history_data)

def main():
    if st.session_state.user_email is None:
        welcome_interface()
    elif st.session_state.is_guest and can_guest_detect():
        st.info(f"You are in Guest mode. Remaining tries: {5 - st.session_state.guest_tries}")
        detection_interface_forguest()
    elif st.session_state.user_email:
        detection_interface()
    st.subheader("Your Upload History")
    df = load_history(st.session_state.user_email or "")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No records found in your history yet.")

if __name__ == "__main__":
    main()
