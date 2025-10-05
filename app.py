import streamlit as st
import datetime
import os

# 1. Page Config (MUST be first)
st.set_page_config(page_title="AVIA - Home", layout="centered")

# 2. Additional Imports
import pyrebase
import pandas as pd
from firebase_config import firebase_config
from firebase_admin_connect import db
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
# 10. Helper: Run Detection on a Sample or File (Dummy)
# -----------------------------------------------------------------------------
def run_detection_on_file(file_path, file_type, logged_in=False):
    detection_result = ""
    confidence = 0.90
    timestamp = datetime.datetime.now().isoformat()
    file_url = None  # local sample => no actual upload

    if logged_in and st.session_state.user_email and not st.session_state.is_guest:
        # Store in Firestore
        detection_data = {
            "uid": st.session_state.uid,
            "user": st.session_state.user_email,
            "filename": os.path.basename(file_path),
            "type": file_type,
            "result": detection_result,
            "confidence": confidence,
            "media_url": file_url,
            "timestamp": timestamp
        }
        try:
            db.collection("detections").add(detection_data)
            st.success("Detection result stored in Firebase!")
        except Exception as firestore_error:
            st.error(f"Error storing detection data: {firestore_error}")
    else:
        st.info("Guest mode or not logged in: no record stored.")

    st.write(f"**Result:** {detection_result} (Confidence: {confidence*100:.2f}%)")

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

# -----------------------------------------------------------------------------
# 12. Main App Logic
# -----------------------------------------------------------------------------
if st.session_state.is_guest:
    # ------------------ GUEST MODE (5-trial limit) ------------------
    st.info(f"You are in Guest mode. Remaining tries: {5 - st.session_state.guest_tries}")

    if not can_guest_detect():
        show_guest_reached_limit()
    else:
        media_source_choice = st.radio("Choose Media Source", ["Sample Media", "Upload Your Own"])
        
        if media_source_choice == "Sample Media":
            st.subheader("Test with Sample Media (Guest)")
            media_type = st.selectbox("Select media type", list(SAMPLE_MEDIA.keys()))
            selected_sample = st.selectbox("Select sample", SAMPLE_MEDIA[media_type], format_func=lambda x: x[0])
            
            if st.button("Analyze Sample"):
                if can_guest_detect():
                    increment_guest_tries()
                    sample_label, sample_path = selected_sample
                    st.write(f"Analyzing **{sample_label}** from path: {sample_path}")
                    run_detection_on_file(sample_path, f"sample/{media_type}", logged_in=False)
                else:
                    show_guest_reached_limit()

        else:
            st.subheader("Guest: Upload Media for Deepfake Detection")
            guest_file = st.file_uploader("Upload image/audio/video", type=["jpg", "png", "jpeg", "wav", "mp3", "mp4"])
            if guest_file is not None:
                st.write("Preview:")
                if guest_file.type.startswith("image"):
                    st.image(guest_file)
                elif guest_file.type.startswith("video"):
                    st.video(guest_file)
                else:
                    st.audio(guest_file)

                    if st.button("Analyze (Guest)"):
                        if can_guest_detect():
                            increment_guest_tries()
                            detection_result = "Real"
                            confidence = 0.50
                            st.success(f"Guest Mode Detection: {detection_result} (Confidence: {confidence:.2f})")
                            st.info("No record is saved for guest users.")
                        else:
                            show_guest_reached_limit()

# ------------------ LOGGED-IN MODE ------------------
elif st.session_state.user_email and not st.session_state.is_guest:
    st.success(f"Welcome, {st.session_state.user_email}!")

    media_source_choice = st.radio("Choose Media Source", ["Sample Media", "Upload Your Own"])

    if media_source_choice == "Sample Media":
        st.subheader("Test with Sample Media (Logged-In)")
        media_type = st.selectbox("Select media type", list(SAMPLE_MEDIA.keys()))
        selected_sample = st.selectbox("Select sample", SAMPLE_MEDIA[media_type], format_func=lambda x: x[0])
        
        if st.button("Analyze Sample"):
            sample_label, sample_path = selected_sample
            st.write(f"Analyzing **{sample_label}** from path: {sample_path}")
            run_detection_on_file(sample_path, f"sample/{media_type}", logged_in=True)

    else:
        st.subheader("Upload Media for Deepfake Detection")
        uploaded_file = st.file_uploader("Upload image/audio/video", type=["jpg", "png", "jpeg", "wav", "mp3", "mp4"])

        if uploaded_file is not None:
            st.write("Preview:")
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file)
            elif uploaded_file.type.startswith("video"):
                st.video(uploaded_file)
            else:
                st.audio(uploaded_file)

            if st.button("Analyze & Upload"):
                local_filename = uploaded_file.name
                with open(local_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Attempt to upload to Firebase Storage
                try:
                    storage.child(f"uploads/{local_filename}").put(local_filename, st.session_state.idToken)
                    file_url = storage.child(f"uploads/{local_filename}").get_url(None)
                    st.write("File uploaded to Storage:", file_url)
                except Exception as upload_error:
                    st.error(f"Error uploading file: {upload_error}")
                    file_url = None

                # Dummy detection logic
                detection_result = "Real"
                confidence = 0.91
                timestamp = datetime.datetime.now().isoformat()

                # Store detection result in Firestore
                detection_data = {
                    "uid": st.session_state.uid,
                    "user": st.session_state.user_email,
                    "filename": local_filename,
                    "type": uploaded_file.type,
                    "result": detection_result,
                    "confidence": confidence,
                    "media_url": file_url,
                    "timestamp": timestamp
                }
                try:
                    db.collection("detections").add(detection_data)
                    st.success("Detection result stored in Firebase!")
                except Exception as firestore_error:
                    st.error(f"Error storing detection data: {firestore_error}")

    # Display user-specific history
    st.subheader("Your Upload History")

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

    history_df = load_history(st.session_state.user_email)
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.info("No records found in your history yet.")

# ------------------ NEITHER LOGGED IN NOR GUEST ------------------
else:
    st.info("Please select an authentication option from the sidebar.")
