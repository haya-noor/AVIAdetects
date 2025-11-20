# Multimodal Deepfake Detection System

A comprehensive deepfake detection system capable of identifying synthetic content across multiple modalities: **video, audio, and images**. This project extends the GenConViT architecture to create a robust multimodal detection framework with enhanced capabilities for diverse media types.

## 🎯 Overview

This project implements a state-of-the-art multimodal deepfake detection system that can analyze and detect synthetic content in videos, images, and audio files. The system leverages deep learning architectures to identify manipulated media with high accuracy, including specialized support for **Urdu language audio detection**.

## 🌟 Inspiration & Background

This project was inspired by and built upon the foundational work of [GenConViT](https://github.com/erprogs/GenConViT). We utilized the original GenConViT model architecture as our starting point but significantly extended its capabilities through:

- **Further training** with custom datasets across multiple modalities
- **Integration of audio detection** models with specialized Urdu language support
- **Integration of image detection** models for standalone image analysis
- **Development of a unified multimodal framework** for comprehensive deepfake detection

The result is a versatile system that goes beyond video-only detection to provide holistic deepfake identification across audio, video, and image domains.

## ✨ Features

### Video Detection
- Frame-by-frame analysis using Vision Transformer architecture
- Temporal consistency evaluation
- Support for various video formats and resolutions
- Real-time and batch processing capabilities

### Audio Detection
- Spectrogram-based analysis for audio deepfake detection
- **Specialized training on Urdu language dataset** for multilingual support
- Detection of voice cloning, speech synthesis, and audio manipulation
- Works seamlessly with both English and Urdu audio content

### Image Detection
- Single-frame deepfake detection
- Face manipulation and GAN-generated image identification
- High-resolution image analysis
- Fast inference for real-time applications

### Multimodal Integration
- Unified pipeline for processing mixed media content
- Cross-modal feature fusion for enhanced accuracy
- Comprehensive reporting across all modalities

## 🛠️ Technologies Used

### Deep Learning Frameworks
- **PyTorch**: Primary deep learning framework for model development and training
- **torchvision**: Image and video preprocessing utilities
- **torchaudio**: Audio processing and feature extraction

### Model Architectures
- **Vision Transformer (ViT)**: Core architecture for image and video analysis (from GenConViT)
- **Convolutional Neural Networks (CNNs)**: Feature extraction layers
- **Transformer Encoders**: Sequence modeling and attention mechanisms
- **Spectrogram-based CNNs**: Audio deepfake detection

### Data Processing
- **OpenCV**: Video frame extraction and image manipulation
- **librosa**: Audio processing and spectrogram generation
- **NumPy & Pandas**: Data manipulation and analysis
- **PIL/Pillow**: Image preprocessing

### Development & Deployment
- **Python 3.8+**: Primary programming language
- **scikit-learn**: Evaluation metrics and data splitting
- **matplotlib & seaborn**: Visualization and result plotting
- **Flask/FastAPI**: API development for model serving (optional)
- **Docker**: Containerization for deployment

### Training Infrastructure
- **CUDA/cuDNN**: GPU acceleration
- **Weights & Biases / TensorBoard**: Experiment tracking
- **Mixed Precision Training**: Efficient model training

## 🏗️ Architecture

### System Overview
```
Input (Video/Audio/Image)
    ↓
Preprocessing Module
    ↓
┌─────────────┬──────────────┬─────────────┐
│   Video     │    Audio     │    Image    │
│  Detection  │  Detection   │  Detection  │
│   Model     │    Model     │    Model    │
└─────────────┴──────────────┴─────────────┘
    ↓
Feature Fusion Layer
    ↓
Classification Head
    ↓
Output (Real/Fake + Confidence Score)
```

### Video Model
- **Base**: GenConViT architecture
- **Input**: Video frames (224x224 RGB)
- **Backbone**: Vision Transformer with convolutional stem
- **Output**: Per-frame and aggregated video-level predictions

### Audio Model
- **Input**: Audio spectrograms (Mel-frequency or STFT)
- **Architecture**: CNN-based feature extractor + Transformer encoder
- **Special Feature**: Trained on Urdu language dataset for multilingual detection
- **Output**: Audio authenticity classification

### Image Model
- **Input**: Single images (224x224 RGB)
- **Architecture**: Modified ViT with specialized detection head
- **Output**: Image-level fake/real classification

## 📊 Dataset

### Video Dataset
- Custom curated dataset combining multiple sources
- Real and synthetic videos from various generation methods
- Diverse facial manipulations and deepfake techniques

### Audio Dataset
- **English Audio**: Standard audio deepfake datasets
- **Urdu Audio**: Custom-collected Urdu language dataset for regional language support
- Voice cloning and TTS-generated samples
- Real recordings from diverse speakers

### Image Dataset
- GAN-generated images (StyleGAN, ProGAN, etc.)
- Face-swap and manipulation datasets
- Real photographs from various sources

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multimodal-deepfake-detection.git
cd multimodal-deepfake-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

```

### Multimodal Analysis
```python
result = detector.detect_multimodal(
    video="path/to/video.mp4",
    audio="path/to/audio.wav",
    image="path/to/image.jpg"
)
print(result)
```

### Command Line Interface
```bash
# Video detection
python detect.py --mode video --input video.mp4

# Audio detection (works with Urdu)
python detect.py --mode audio --input audio.wav

# Image detection
python detect.py --mode image --input image.jpg

# Multimodal detection
python detect.py --mode multimodal --video video.mp4 --audio audio.wav
```

## 👥 Team

This project was completed as a **Bachelor's Thesis** by:

- **Haya Noor** - [GitHub](https://github.com/haya-noor) 
- **Lailoma** - [GitHub](https://github.com/lailomanoor) 
- **Itba** - [GitHub](https://github.com/ItbaMalahat) 

### Contributions
- **Model Development**: All team members
- **Video Module**: [Haya Noor]
- **Audio Module & Urdu Dataset**: [Itba Malahat]
- **Image Module**: [Lailoma Noor]
- **Integration & Testing**: Collaborative effort

## 🙏 Acknowledgments

- **GenConViT Project**: Special thanks to the [GenConViT repository](https://github.com/erprogs/GenConViT) for providing the foundational model architecture that inspired this work
- Our thesis advisors and mentors for their guidance
- The open-source community for various tools and libraries
- Dataset contributors and researchers in the deepfake detection community

---

**Note**: This is an academic research project. Models and results should be validated for production use cases. Always consider ethical implications when deploying deepfake detection systems.
