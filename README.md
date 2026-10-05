# AVIA — Audio-Visual Integrity Analyzer (Multimodal Deepfake Detection)
A Streamlit web application that detects deepfakes across video, audio, and image media. Built on top of the GenConViT architecture for video, with dedicated models for audio (including Urdu-language support) and image deepfake detection, plus Firebase-backed accounts, guest mode, and per-user detection history.

## Overview
AVIA lets a user sign up / log in (or continue as a guest) and upload a video, audio clip, or image to get a Real/Fake prediction with a confidence score. Logged-in users' results are saved to Firestore and shown back to them as detection history.

## Features
 - Video Detection — frame sampling + face extraction, classified with a GenConViT (ViT + CNN) model.
 - Audio Detection — MFCC-feature spectrogram analysis via a Keras/TensorFlow model (trained with Urdu-language audio support).
 - Image Detection — ConvNeXt-based real/fake classifier, plus a second-stage classifier that identifies the likely generation technique (e.g. StyleGAN2, ProGAN,   Stable Diffusion, StarGAN, Denoising Diffusion GAN) for images flagged as fake.
 - Accounts & Guest Mode — Firebase Authentication for login/signup, a no-login guest mode, and cookie-based session persistence.
 - Detection History — results are written to Firestore and listed back to the logged-in user.

## Technologies Used
- Streamlit — web UI / app framework
- PyTorch + torchvision — video (GenConViT) and image (ConvNeXt) models
- TensorFlow / Keras — audio deepfake model
- librosa — audio feature extraction (MFCC)
- OpenCV, dlib, face_recognition, decord — video frame/face extraction
- timm, albumentations — model backbones and image augmentation
- Firebase (pyrebase, firebase-admin) — authentication, Firestore history, storage
- streamlit-cookies-manager — persisting login across sessions

 ## Project Structure
 ```
AVIAdetects/
 ├── common_firebase.py        # Main Streamlit app — run this file
 ├── convnext_image.py         # ConvNeXt model definition (image detection)
 ├── firebase_config.py        # Firebase web app config (pyrebase)
 ├── firebase_admin_connect.py # Firebase Admin SDK init (Firestore)
 ├── firestore.indexes.json    # Firestore index definitions
 ├── requirements.txt
 ├── model/                    # GenConViT video model
 │   ├── genconvit.py
 │   ├── genconvit_ed.py
 │   ├── genconvit_vae.py
 │   ├── model_embedder.py
 │   ├── pred_func.py          # video inference pipeline (face extraction, prediction)
 │   ├── config.py / config.yaml
 ├── dataset/
 │   └── loader.py             # data normalization/augmentation helpers used by model/pred_func.py
 └── weight/                   # put GenConViT .pth weights here (not committed, see below)
```

## Installation & Setup
# Prerequisites
- Python 3.8+
- CUDA-capable GPU recommended (CPU also works, just slower for video)
- cmake and a C++ build toolchain installed on your system before pip install (required to build dlib)

1. Clone the repository
```
git clone https://github.com/haya-noor/AVIAdetects.git
cd AVIAdetects
```

2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Add the model weight files
None of the trained model weights are committed to this repo (they're large binaries, excluded via .gitignore). You need to obtain/place them yourself:

| File | Used for | Location |
|---|---|---|
| `genconvit_ed_inference.pth` | Video detection (encoder-decoder) | `weight/` |
| `genconvit_vae_inference.pth` | Video detection (VAE) | `weight/` |
| `my_model.h5` | Audio detection | project root |
| `convnext_tiny_1k_224_ema_image.pth` | Image real/fake detection | project root |
| `checkpoint_epoch_20 (2).pth` | Image real/fake classifier checkpoint | project root |
| `convnext2_epoch_20.pth` | Image generation-technique classifier | project root |

5. Configure Firebase
- firebase_config.py holds the Firebase web app config (already present in this repo).
- firebase_admin_connect.py needs a serviceAccountKey.json (Firebase Admin SDK service account) in the project root — this is not committed (it's a secret). Generate one from your Firebase project's Settings → Service Accounts, and save it as serviceAccountKey.json at the repo root.

6. Run the app
```
streamlit run common_firebase.py
```
This opens the app in your browser (default http://localhost:8501). From the welcome screen you can continue as a Guest, or Login / Sign Up to get persistent history.

## Dataset
- Video: curated real + synthetic videos covering multiple deepfake generation methods.
- Audio: standard audio-deepfake datasets plus a custom-collected Urdu language dataset for regional-language support.
- Image: GAN-generated images (StyleGAN2, ProGAN, StarGAN, etc.) and diffusion-generated images, alongside real photographs.

## Acknowledgments
- GenConViT: this project's video model is built on and extends the GenConViT architecture.
- The open-source community and dataset contributors in the deepfake-detection research space.
- Our thesis advisors for their guidance.

## Team
This project was completed as a Bachelor's Thesis by:

Haya Noor — [GitHub](https://github.com/haya-noor) — Video Module
Lailoma — [GitHub](https://github.com/lailomanoor) — Image Module
Itba — [GitHub](https://github.com/ItbaMalahat) — Audio Module & Urdu Dataset

Integration, Firebase backend, and testing were a collaborative effort.
