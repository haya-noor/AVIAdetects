# new_common.py  – AVIA | Cinzel logo, dark canvas, single-card after upload
# guest mode, welcome page, login/signup & history

import streamlit as st, datetime, io, os, tempfile, numpy as np
from PIL import Image
import tensorflow as tf, librosa, torch, torch.nn as nn
from torchvision import transforms
from model.pred_func import load_genconvit, df_face, pred_vid, is_video, real_or_fake
from model.config import load_config
import pyrebase
from firebase_config import firebase_config
from firebase_admin_connect import db

# 1. Page Config (MUST be first)
st.set_page_config(page_title="AVIA - Home", layout="centered")

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

# A counter for how many detection attempts the guest has made
if "guest_tries" not in st.session_state:
    st.session_state.guest_tries = 0

# -----------------------------------------------------------------------------
# 5. Cookies Manager Initialization
# -----------------------------------------------------------------------------
cookies = EncryptedCookieManager(prefix="aviadetects_", password="somestrongpassword123")
if not cookies.ready():
    st.stop()  # wait until cookies manager is ready

# -----------------------------------------------------------------------------
# 6. Pyrebase Initialization (Firebase Auth & Storage)
# -----------------------------------------------------------------------------
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
storage = firebase.storage()
for k in ("user_email","idToken","uid","guest_count","show_auth","viewing_history"):
    st.session_state.setdefault(k, None)

# -----------------------------------------------------------------------------
# 7. Load Auth Info from Cookies
# -----------------------------------------------------------------------------
def load_cookies():
    if not st.session_state.user_email and "user_email" in cookies and cookies.get("user_email"):
        if cookies.get("user_email").strip():
            st.session_state.user_email = cookies.get("user_email")
    if not st.session_state.idToken and "idToken" in cookies and cookies.get("idToken"):
        if cookies.get("idToken").strip():
            st.session_state.idToken = cookies.get("idToken")
    if not st.session_state.uid and "uid" in cookies and cookies.get("uid"):
        if cookies.get("uid").strip():
            st.session_state.uid = cookies.get("uid")

load_cookies()


# -----------------------------------------------------------------------------
# 8. Sidebar: Dynamic Authentication Menu
# -----------------------------------------------------------------------------
st.sidebar.header("Authentication")

def do_logout():
    st.session_state.user_email = None
    st.session_state.idToken = None
    st.session_state.uid = None
    st.session_state.is_guest = False
    st.session_state.guest_tries = 0
    cookies["user_email"] = ""
    cookies["idToken"] = ""
    cookies["uid"] = ""
    cookies.save()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)

def handle_login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.is_guest = False
        st.session_state.user_email = email
        st.session_state.idToken = user["idToken"]
        st.session_state.uid = user["localId"]
        st.session_state.guest_tries = 0  # reset
        cookies["user_email"] = email
        cookies["idToken"] = user["idToken"]
        cookies["uid"] = user["localId"]
        cookies.save()
        st.sidebar.success("Login successful!")
    except Exception as e:
        st.sidebar.error(f"Login error: {e}")

def handle_signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state.is_guest = False
        st.session_state.user_email = email
        st.session_state.idToken = user["idToken"]
        st.session_state.uid = user["localId"]
        st.session_state.guest_tries = 0  # reset
        cookies["user_email"] = email
        cookies["idToken"] = user["idToken"]
        cookies["uid"] = user["localId"]
        cookies.save()
        st.sidebar.success("Signup successful!")
    except Exception as e:
        st.sidebar.error(f"Signup error: {e}")

def handle_guest():
    # Clear any existing login
    st.session_state.user_email = None
    st.session_state.idToken = None
    st.session_state.uid = None
    cookies["user_email"] = ""
    cookies["idToken"] = ""
    cookies["uid"] = ""
    cookies.save()
    st.session_state.is_guest = True
    st.session_state.guest_tries = 0
    st.sidebar.success("You are now a Guest user!")

# Decide what to show in the sidebar based on user state
if st.session_state.user_email and not st.session_state.is_guest:
    # Logged in => only show logout
    if st.sidebar.button("Logout"):
        do_logout()

elif st.session_state.is_guest:
    # Guest => show only login & signup
    st.sidebar.write("Currently in Guest Mode")
    guest_login_email = st.sidebar.text_input("Email for Login").strip()
    guest_login_pw = st.sidebar.text_input("Password for Login", type="password")
    if st.sidebar.button("Login as User"):
        handle_login(guest_login_email, guest_login_pw)

    guest_signup_email = st.sidebar.text_input("Email for Signup").strip()
    guest_signup_pw = st.sidebar.text_input("Password for Signup", type="password")
    if st.sidebar.button("Create Account"):
        handle_signup(guest_signup_email, guest_signup_pw)

else:
    # Not logged in, not guest => show all three
    st.sidebar.write("Select an option below:")
    new_login_email = st.sidebar.text_input("Email for Login").strip()
    new_login_pw = st.sidebar.text_input("Password for Login", type="password")

    if st.sidebar.button("Login"):
        handle_login(new_login_email, new_login_pw)

    new_signup_email = st.sidebar.text_input("Email for Signup").strip()
    new_signup_pw = st.sidebar.text_input("Password for Signup", type="password")

    if st.sidebar.button("Signup"):
        handle_signup(new_signup_email, new_signup_pw)

    if st.sidebar.button("Proceed as Guest"):
        handle_guest()

# -----------------------------------------------------------------------------
# 9. Sample Media Dictionary
# -----------------------------------------------------------------------------
SAMPLE_MEDIA = {
    "Audio": [
        ("Real Audio", "samples/real_audio.wav"),
        ("Fake Audio", "samples/fake_audio.wav")
    ],
    "Image": [
        ("Real Image", "samples/real_image.jpg"),
        ("Fake Image", "samples/fake_image.jpg")
    ],
    "Video": [
        ("Real Video", "samples/real_video.mp4"),
        ("Fake Video", "samples/fake_video.mp4")
    ]
}

# -----------------------------------------------------------------------------
# 11. Guest Tries Helper
# -----------------------------------------------------------------------------
def can_guest_detect():
    """Returns True if guest can still run detection (under 5 tries)."""
    return st.session_state.guest_tries < 5

def increment_guest_tries():
    st.session_state.guest_tries += 1

def show_guest_reached_limit():
    st.warning("You have reached the maximum of 5 guest detections. Please log in or sign up for unlimited usage.")

# ───────────────────────────  THEME & STYLE  ────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600&display=swap');
    :root{--primary:#4B6FE7;--accent:#6E35E8;--canvas:#0D1A34;}
    [data-testid="stApp"]>div:first-child{max-width:85%!important;margin:0 auto!important;}

    html,body,[data-testid="stApp"]{
        background:radial-gradient(circle at 10% 10%,#13224A 0%,var(--canvas) 60%) fixed;
    }
    html,body,div,span,p,label,input,textarea{
        font-size:20px!important; color:#ECEFF4!important;
    }
    h1{
        font:600 7rem 'Cinzel',serif!important;
        letter-spacing:.07em; color:var(--primary)!important; margin-bottom:.2em;
    }
    /* primary buttons */
    button[kind="primary"], button{
        font-size:1.3rem!important; font-weight:600!important; color:#fff!important;
        background:linear-gradient(90deg,var(--primary),var(--accent))!important;
        border:none!important; border-radius:6px!important;
        box-shadow:0 2px 4px rgba(0,0,0,.15)!important;
    }
    button[kind="primary"]:hover, button:hover{
        background:linear-gradient(90deg,var(--accent),var(--primary))!important;
    }
    /* card */
    .card{
        background:rgba(255,255,255,.05);
        padding:1.5rem; border-radius:12px;
        box-shadow:0 2px 8px rgba(0,0,0,.35);
    }
    /* transparent dropdown + uploader */
    .stSelectbox, .stFileUploader>div{
        background:rgba(255,255,255,.04)!important;
        border:1px solid rgba(255,255,255,.15)!important;
        border-radius:6px!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ─────────────────────────────  HELPERS  ────────────────────────────────
class FixedBN(tf.keras.layers.BatchNormalization):
    def __init__(self,*a,**kw):
        kw["axis"]=kw.get("axis",-1) if isinstance(kw.get("axis"),list) else kw.get("axis")
        super().__init__(*a,**kw)
tf.keras.utils.get_custom_objects()['BatchNormalization']=FixedBN

@st.cache_resource
def load_audio_model():
    return tf.keras.models.load_model(
        "my_model.h5", custom_objects={"BatchNormalization":FixedBN}, compile=False
    )

def extract_features(b,sr=16000,n_mfcc=40,max_len=500):
    y,_ = librosa.load(io.BytesIO(b),sr=sr)
    m   = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
    m   = np.pad(m,((0,0),(0,max_len-m.shape[1])),mode="constant") if m.shape[1]<max_len else m[:,:max_len]
    return m.reshape(1,*m.shape,1)

def convnext_image_model():
    from convnext_image import ConvNeXt_image
    mdl = ConvNeXt_image()
    mdl.load_state_dict(torch.load("convnext_tiny_1k_224_ema_image.pth",map_location="cpu")["model"])
    return mdl

def convnext_image_model_tech():
    m=convnext_image_model(); m.head=nn.Linear(m.head.in_features,5); return m

def tf_img(img):
    t=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return t(img).unsqueeze(0)

def test_single(model,img,dev):
    model.eval()
    with torch.no_grad():
        out=model(tf_img(img).to(dev))
        p  = torch.softmax(out,1).cpu().numpy()
    return ("Fake","Real")[out.argmax(1).item()], p[0].max()*100

def test_tech(model,img,dev):
    model.eval()
    with torch.no_grad():
        out=model(tf_img(img).to(dev))
        p  = torch.softmax(out,1).cpu().numpy()
    tech={0:"denoising_diffusion_gan",1:"stable_diffusion",2:"star_gan",3:"stylegan2",4:"pro_gan"}[out.argmax(1).item()]
    return tech, p[0].max()*100

def remove_key_recursively(d,key):
    if isinstance(d,dict):
        d.pop(key,None)
        for v in d.values(): remove_key_recursively(v,key)
    elif isinstance(d,list):
        for i in d: remove_key_recursively(i,key)

def predict_video_file(path,model,n=15,net="genconvit",fp16=False):
    if not is_video(path): return None,None
    df=df_face(path,n,net); df=df.half() if fp16 else df
    y,conf=pred_vid(df,model) if len(df)>=1 else (0,0)
    return y,conf

import datetime

def store_result(tp, filename, result, confidence, local_path=None):
    """
    Uploads `local_path` to Firebase Storage (if provided), 
    then stores a detection record in Firestore.
    
    Args:
      tp (str): media type, e.g. "Image", "Audio", "Video"
      filename (str): the name under which to store the file/data
      result (str): "Real" or "Fake"
      confidence (float): confidence score (0–100)
      local_path (str, optional): local filesystem path to upload
    """
    file_url = None

    # 1️⃣ Optional upload to Storage
    if local_path:
        try:
            # storage is assumed to be your pyrebase.storage() instance
            storage.child(f"uploads/{filename}").put(local_path, st.session_state.idToken)
            file_url = storage.child(f"uploads/{filename}").get_url(None)
            st.write("File uploaded to Storage:", file_url)
        except Exception as upload_error:
            st.error(f"Error uploading file: {upload_error}")
            file_url = None

    # 2️⃣ Always attempt to store the detection record
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

# ─────────────────────────────  UI PARTS  ───────────────────────────────
def auth_sidebar():
    st.sidebar.header("Authentication")
    act   = st.sidebar.radio("Action",["Login","Sign Up"])
    email = st.sidebar.text_input("Email")
    pwd   = st.sidebar.text_input("Password",type="password")
    if act=="Login" and st.sidebar.button("Login"):
        try:
            u=auth.sign_in_with_email_and_password(email,pwd)
            st.session_state.update(user_email=email,idToken=u["idToken"],uid=u["localId"],show_auth=False); st.rerun()
        except Exception as e: st.error(f"Login failed: {e}")
    if act=="Sign Up" and st.sidebar.button("Sign Up"):
        try:
            u=auth.create_user_with_email_and_password(email,pwd)
            st.session_state.update(user_email=email,idToken=u["idToken"],uid=u["localId"],show_auth=False); st.rerun()
        except Exception as e: st.error(f"Signup failed: {e}")

def welcome_interface():
    st.markdown("<h1 style='text-align:center'>AVIA</h1>"
                "<p style='text-align:center;margin-top:-1rem;font-size:1.3rem'>"
                "Audio-Visual Integrity Analyzer&nbsp;|&nbsp;Deepfake Detection</p><hr>",
                unsafe_allow_html=True)
    _,mid,_ = st.columns([1,2,1])
    with mid:
        l,r = st.columns(2)
        if l.button("🔓 Guest Mode"):
            st.session_state.update(user_email="guest",guest_count=5,show_auth=False); st.rerun()
        if r.button("🔐 Login / Sign Up"): st.session_state.show_auth=True
    if st.session_state.show_auth: auth_sidebar()

def detection_interface():
    st.title("AVIA")
    mode = st.selectbox("Choose Detection Type",("Audio","Image","Video"),index=0)

    # 1️⃣ upload widget (always visible, translucent background)
    if mode=="Audio":
        uploaded = st.file_uploader("Upload Audio",type=["wav","mp3","ogg"])
    elif mode=="Image":
        uploaded = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])
    else:
        uploaded = st.file_uploader("Upload Video",type=["avi","mp4","mov","mpg","mpeg"])

    # 2️⃣ card shows ONLY once a file is selected
    if uploaded:
        st.markdown("<div class='card'>",unsafe_allow_html=True)

        # AUDIO
        if mode=="Audio":
            if st.button("Analyze Audio"):
                pred = load_audio_model().predict(extract_features(uploaded.getvalue()))[0][0]
                lbl,conf = ("Fake",pred*100) if pred>0.5 else ("Real",(1-pred)*100)
                st.success(f"{lbl} • {conf:.2f}%"); store_result("Audio",uploaded.name,lbl,conf)

        # IMAGE
        elif mode=="Image":
            img=Image.open(uploaded).convert("RGB"); st.image(img)
            if st.button("Analyze Image"):
                dev=torch.device("cpu")
                mdl=convnext_image_model().to(dev)
                mdl.load_state_dict(torch.load("checkpoint_epoch_20 (2).pth",map_location=dev)["model_state_dict"])
                lbl,conf=test_single(mdl,img,dev); st.success(f"{lbl} • {conf:.2f}%"); store_result("Image",uploaded.name,lbl,conf)
                if lbl=="Fake":
                    mdl2=convnext_image_model_tech().to(dev)
                    mdl2.load_state_dict(torch.load("convnext2_epoch_20.pth",map_location=dev)["model_state_dict"])
                    tech,tconf=test_tech(mdl2,img,dev); st.info(f"Technology: {tech} ({tconf:.2f}%)")

        # VIDEO
        else:
            st.video(uploaded)
            frames = st.number_input("Frames to sample",1,60,15)
            fp16   = st.checkbox("Enable FP16",True) if torch.cuda.is_available() else False
            if st.button("Analyze Video"):
                tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tmp.write(uploaded.getvalue()); tmp.close()
                cfg=load_config(); remove_key_recursively(cfg,"pretrained_cfg")
                mdl=load_genconvit(cfg,net="genconvit",ed_weight="genconvit_ed_inference",
                                   vae_weight="genconvit_vae_inference",fp16=fp16)
                if fp16: mdl=mdl.half()
                y,conf=predict_video_file(tmp.name,mdl,frames,fp16=fp16); os.remove(tmp.name)
                if y is not None:
                    pred=real_or_fake(y); conf=1-conf if pred=="REAL" else conf
                    st.success(f"{pred} • {conf*100:.2f}%"); store_result("Video",uploaded.name,pred,conf*100)
                else:
                    st.error("Prediction failed")

        st.markdown("</div>",unsafe_allow_html=True)  # close card





#----------------------------------------------------------------------------------
def load_history(user_email):
        docs = db.collection("detections").where("user", "==", user_email).stream()
        history_data = []
        for doc in docs:
            record = doc.to_dict()
            history_data.append({
                "UID": record.get("uid", ""),
                "User": record.get("user", ""),
                "File": record.get("filename", ""),
                "Type": record.get("type", ""),
                "Result": record.get("result", ""),
                "Confidence": record.get("confidence", ""),
                "Media URL": record.get("media_url", ""),
                "Timestamp": record.get("timestamp", "")
            })
        return pd.DataFrame(history_data)

# ─────────────────────────────  ROUTING  ───────────────────────────────
def main():
    if st.session_state.user_email is None:
        welcome_interface()
        
    # elif st.session_state.user_email == "guest" and st.session_state.guest_count > 0:
    #     # ------------------ GUEST MODE (5-trial limit) ------------------
    #     st.info(f"You are in Guest mode. Remaining tries: {5 - st.session_state.guest_tries}")

    #     if not can_guest_detect():
    #         show_guest_reached_limit()
    #     else:
    #         detection_interface()

   if st.session_state.user_email and st.session_state.user_email != "guest":
    # Logged‐in user sidebar actions
    with st.sidebar:
        if st.button("📂 View My History"):
            st.experimental_rerun()
        if st.button("🚪 Logout"):
            st.session_state.update(
                user_email=None,
                idToken=None,
                uid=None,
                show_auth=None
            )
            st.experimental_rerun()

    # Main detection interface
    detection_interface()

    # Then display the user’s history below
    history_df = load_history(st.session_state.user_email)
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.info("No records found in your history yet.")

else:
    # Guest limit reached or no auth
    st.warning("Guest limit reached. Please sign up or log in.")
    st.session_state.user_email = None
    welcome_interface()

if __name__ == "__main__":
    main()
