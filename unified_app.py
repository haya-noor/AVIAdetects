import streamlit as st
import os
import io
import numpy as np
from PIL import Image

# --- AUDIO IMPORTS ---
import librosa
import tensorflow as tf

# --- IMAGE IMPORTS ---
import torch
from torchvision import transforms
from convnext_image import ConvNeXt

# ----------------------------
# DEVICE CONFIGURATION
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# AUDIO PATCHES & HELPERS
# ----------------------------

# Custom BatchNormalization to fix axis deserialization issues
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

# Register globally
tf.keras.utils.get_custom_objects()['BatchNormalization'] = FixedBatchNormalization

@st.cache_resource
def load_audio_model():
    try:
        model = tf.keras.models.load_model(
            "updated_model.keras",  # or your .h5 path
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
# IMAGE HELPERS
# ----------------------------

def ConvNeXt_model():
    model_conv = ConvNeXt()
    state_dict = torch.load('convnext_tiny_1k_224_ema.pth', map_location=device)
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
# MAIN STREAMLIT APP
# ----------------------------

st.set_page_config(page_title="AVIA - Unified Deepfake Detection", layout="centered")
st.title("🎭 AVIA: Unified Deepfake Detection")
mode = st.selectbox("Select Detection Mode", ["Audio", "Image"])

# ====================
# AUDIO MODE
# ====================
if mode == "Audio":
    st.subheader("🎧 Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Detect Audio Deepfake"):
            model = load_audio_model()
            if model:
                features = extract_features_from_audio(uploaded_file.getvalue())
                if features is not None:
                    prediction = model.predict(features)
                    confidence = prediction[0][0]
                    is_deepfake = confidence > 0.5

                    st.subheader("Detection Result")
                    if is_deepfake:
                        st.error(f"🚨 Deepfake Detected (Confidence: {confidence * 100:.2f}%)")
                    else:
                        st.success(f"✅ Real Audio (Confidence: {(1 - confidence) * 100:.2f}%)")
                    st.progress(float(confidence))

# ====================
# IMAGE MODE
# ====================
elif mode == "Image":
    st.subheader("🖼 Upload Image Frame")
    uploaded_image = st.file_uploader("Choose an image frame", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        checkpoint_path = 'checkpoint_epoch_20 (2).pth'

        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found at {checkpoint_path}")
        else:
            try:
                model = ConvNeXt_model()
                checkpoint = torch.load(checkpoint_path, map_location=device)

                if 'model_state_dict' not in checkpoint:
                    st.error("Invalid checkpoint format.")
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    predicted_label, confidence_score = test_single_image(model, image, device)

                    st.subheader("Prediction Results")
                    st.write(f"*Predicted Class:* {predicted_label}")
                    st.write(f"*Confidence Score:* {confidence_score:.2f}%")
            except Exception as e:
                st.error(f"Error during model inference: {e}")