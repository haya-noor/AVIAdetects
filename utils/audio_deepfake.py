# import tensorflow as tf
# import numpy as np

# # Custom BatchNormalization to fix axis deserialization issues
# class FixedBatchNormalization(tf.keras.layers.BatchNormalization):
#     def __init__(self, *args, **kwargs):
#         axis = kwargs.get("axis")
#         if isinstance(axis, list) and len(axis) > 0:
#             kwargs["axis"] = axis[0]
#         super(FixedBatchNormalization, self).__init__(*args, **kwargs)
    
#     @classmethod
#     def from_config(cls, config):
#         axis = config.get("axis")
#         if isinstance(axis, list) and len(axis) > 0:
#             config["axis"] = axis[0]
#         return cls(**config)
    
#     def get_config(self):
#         config = super(FixedBatchNormalization, self).get_config()
#         axis = config.get("axis")
#         if isinstance(axis, list) and len(axis) > 0:
#             config["axis"] = axis[0]
#         return config

# # Register the custom BatchNormalization globally.
# tf.keras.utils.get_custom_objects()['BatchNormalization'] = FixedBatchNormalization

# model_path = r'updated_model.keras'  # adjust as needed

# try:
#     # Try loading with compile=False
#     model = tf.keras.models.load_model(
#         model_path,
#         custom_objects={"BatchNormalization": FixedBatchNormalization},
#         compile=False
#     )
#     print("Model loaded successfully.")
    
#     # Create a dummy input matching expected shape (1, 40, 500, 1)
#     dummy_input = np.zeros((1, 40, 500, 1), dtype=np.float32)
#     prediction = model.predict(dummy_input)
#     print("Prediction on dummy input:", prediction)
# except Exception as e:
#     print("Error loading or using the model:", e)

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io

# --- Patch for BatchNormalization axis deserialization ---
_original_bn_from_config = tf.keras.layers.BatchNormalization.from_config

def patched_bn_from_config(config):
    if "axis" in config and isinstance(config["axis"], list):
        config["axis"] = config["axis"][0]
    return _original_bn_from_config(config)

tf.keras.layers.BatchNormalization.from_config = patched_bn_from_config
# --- End Patch ---

# --- Patch for Orthogonal initializer deserialization ---
_original_deserialize = tf.keras.initializers.deserialize

def patched_initializer_deserialize(identifier, custom_objects=None):
    if isinstance(identifier, dict) and identifier.get("class_name") == "Orthogonal":
        config = identifier.get("config", {})
        config.pop("shared_object_id", None)
        config.pop("registered_name", None)
        return tf.keras.initializers.Orthogonal(**config)
    return _original_deserialize(identifier, custom_objects)

tf.keras.initializers.deserialize = patched_initializer_deserialize
# --- End Patch ---

# Function to extract MFCC features from uploaded audio
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

def main():
    st.title("Audio Deepfake Detection")
    st.write("Upload an audio file to detect if it is real or deepfake.")

    @st.cache_resource
    def load_model():
        try:
            # Load the .h5 model file.
            model = tf.keras.models.load_model("my_model.h5")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "ogg"],
        help="Upload an audio file to check if it's a deepfake"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Detect Deepfake"):
            if model is not None:
                features = extract_features_from_audio(uploaded_file.getvalue())
                if features is not None:
                    prediction = model.predict(features)
                    confidence = prediction[0][0]
                    is_deepfake = confidence > 0.5

                    st.subheader("Detection Results")
                    if is_deepfake:
                        st.error(f"🚨 Deepfake Detected (Confidence: {confidence*100:.2f}%)")
                    else:
                        st.success(f"✅ Real Audio (Confidence: {(1-confidence)*100:.2f}%)")
                    st.progress(float(confidence))
            else:
                st.error("Model could not be loaded. Please check the model file.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a deep learning model to detect audio deepfakes. "
        "It analyzes the audio file's MFCC features to classify if the audio is real or synthetic."
    )
    st.sidebar.header("How it Works")
    st.sidebar.write(
        "1. Upload an audio file\n"
        "2. Click 'Detect Deepfake'\n"
        "3. Get instant results with confidence percentage"
    )

if __name__ == "__main__":
    main()
