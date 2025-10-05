import os
import torch
import streamlit as st
import tempfile
from model.pred_func import *
from model.config import load_config

@st.cache_resource
def load_model(net="genconvit", ed_weight="genconvit_ed_inference", vae_weight="genconvit_vae_inference", fp16=False):
    config = load_config()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    if fp16:
        model = model.half()
    return model

def predict_video_streamlit(model, video_path, num_frames=15, net=None, fp16=False):
    try:
        if is_video(video_path):
            df = df_face(video_path, num_frames, net)
            if fp16:
                df = df.half()  # Convert to FP16 if enabled
            if len(df) >= 1:
                # Assume pred_vid returns (prediction, confidence)
                y, confidence = pred_vid(df, model)
            else:
                y = torch.tensor(0).item()  # Default prediction
                confidence = 0.0

            label = real_or_fake(y)
            return label, confidence
        else:
            st.error("Invalid video file. Please upload a valid video.")
            return None, None
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None

def main():
    st.title("Deepfake Detection")
    st.write("Upload a video file for prediction.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # Read the file bytes once and store them in a variable.
        file_bytes = uploaded_file.read()

        # Display the uploaded video using the file bytes.
        st.video(file_bytes)
        
        # Extract the file extension from the uploaded file name.
        ext = os.path.splitext(uploaded_file.name)[1]
        
        # Write the file bytes to a temporary file with the proper suffix.
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_bytes)
            video_path = temp_file.name

        # Load the model (this is cached).
        model = load_model()

        # Run prediction on the video file.
        label, confidence = predict_video_streamlit(model, video_path, num_frames=15, net="genconvit", fp16=False)

        if label is not None:
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence Score:** {confidence * 100:.2f}%")  # Display as percentage.

if __name__ == "__main__":
    main()
