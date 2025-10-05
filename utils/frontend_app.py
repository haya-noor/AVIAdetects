import streamlit as st
from PIL import Image
import io

# --- Set up professional styling ---
st.set_page_config(page_title="AVIA - Deepfake Detection", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global styles */
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
    /* Result box styling */
    .result-box {
        border-radius: 10px;
        padding: 20px;
        background: rgba(56, 61, 74, 0.8);
        border: 1px solid #30363d;
        margin-top: 20px;
        text-align: center;
    }
    /* Progress bar gradient */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #238636);
    }
    </style>
""", unsafe_allow_html=True)

# --- Page Title & Description ---
st.title("🤖 AVIA: Deepfake Detection")
st.write("Experience advanced AI-powered detection for deepfakes in audio and images with AVIA.")

# --- Sidebar Information ---
st.sidebar.title("About AVIA")
st.sidebar.info(
    "AVIA uses state-of-the-art deep learning models to analyze media integrity. "
    "Our system detects whether the input audio or image is genuine or a deepfake, "
    "bringing cutting-edge AI to your fingertips."
)
st.sidebar.title("How It Works")
st.sidebar.markdown("""
1. **Select** detection mode: Audio or Image.  
2. **Upload** your media file.  
3. **Click** the detection button.  
4. **View** the result along with confidence score.
""")

# --- Detection Mode Switch ---
detection_mode = st.radio("Select Detection Type", ["Audio Detection", "Image Detection"], index=0)

# --- Audio Detection Section ---
if detection_mode == "Audio Detection":
    st.header("🎙️ Audio Deepfake Detection")
    audio_file = st.file_uploader("Upload Audio (supported: wav, mp3, ogg)", type=["wav", "mp3", "ogg"])
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        if st.button("Analyze Audio"):
            # --- Replace dummy logic with your audio model inference ---
            # Example: features = extract_features_from_audio(audio_file.getvalue())
            #          prediction = audio_model.predict(features)
            #          fake_confidence = prediction[0][0]
            fake_confidence = 0.65  # dummy value
            
            if fake_confidence > 0.5:
                result = f"<p style='color:#ff7b72; font-size:1.2rem;'>Deepfake Audio Detected (Confidence: {fake_confidence*100:.2f}%)</p>"
            else:
                result = f"<p style='color:#3fb950; font-size:1.2rem;'>Real Audio (Confidence: {(1-fake_confidence)*100:.2f}%)</p>"
            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
            st.progress(fake_confidence)

# --- Image Detection Section ---
elif detection_mode == "Image Detection":
    st.header("🖼️ Image Deepfake Detection")
    image_file = st.file_uploader("Upload Image (supported: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            # --- Replace dummy logic with your image model inference ---
            # Example: predicted_label, confidence_score = test_single_image(model, image, device)
            fake_confidence = 0.32  # dummy value
            
            if fake_confidence > 0.5:
                result = f"<p style='color:#ff7b72; font-size:1.2rem;'>Deepfake Image Detected (Confidence: {fake_confidence*100:.2f}%)</p>"
            else:
                result = f"<p style='color:#3fb950; font-size:1.2rem;'>Real Image (Confidence: {(1-fake_confidence)*100:.2f}%)</p>"
            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
            st.progress(fake_confidence)
