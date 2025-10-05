import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from convnext_image import ConvNeXt

# ----------------------------
# Device Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# ConvNeXt Model Function
# ----------------------------
def ConvNeXt_model():
    model_conv = ConvNeXt()
    state_dict = torch.load('convnext_tiny_1k_224_ema.pth', map_location=device)
    model_conv.load_state_dict(state_dict["model"])
    return model_conv

# ----------------------------
# Image Transformation for Single Image
# ----------------------------
def transform_single_image(image):
    """
    Preprocess a single image before passing it to the model.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ----------------------------
# Single Image Testing Function
# ----------------------------
def test_single_image(model, image, device, temperature=2.0):
    """
    Test the model on a single image and return the predicted class with confidence percentage.
    """
    model.eval()
    image_tensor = transform_single_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)

        # Apply Temperature Scaling to Softmax
        probabilities = torch.nn.functional.softmax(output / temperature, dim=1).cpu().numpy()
        predicted_class = output.argmax(dim=1).item()
        confidence_score = probabilities[0][predicted_class] * 100  # Convert to percentage

    # Map class index to label
    class_labels = {0: "Fake", 1: "Real"}
    predicted_label = class_labels.get(predicted_class, "Unknown")

    return predicted_label, confidence_score

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's **Fake** or **Real**.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load Model
    checkpoint_path = 'checkpoint_epoch_20 (2).pth'

    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint file not found at {checkpoint_path}")
    else:
        model = ConvNeXt_model()
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Ensure checkpoint is correctly formatted
            if 'model_state_dict' not in checkpoint:
                st.error("Checkpoint does not contain 'model_state_dict' key.")
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)

                # Run inference on the uploaded image
                predicted_label, confidence_score = test_single_image(model, image, device)

                # Display results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Class:** {predicted_label}")
                st.write(f"**Confidence Score:** {confidence_score:.2f}%")

        except Exception as e:
            st.error(f"Error loading model checkpoint: {e}")
