import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Haar cascade face detector (pure-Python via OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    model = GenConViT(
        config,
        ed=ed_weight,
        vae=vae_weight,
        net=net,
        fp16=fp16
    )
    model.to(device)
    model.eval()
    if fp16:
        model.half()
    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        # Convert to grayscale for Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detections = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, w, h) in detections:
            if count < len(frames):
                # Crop and resize face region
                face_img = frame[y : y + h, x : x + w]
                face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                temp_face[count] = face_img
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))
    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)
    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item() if mean_val[0] > mean_val[1] else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()


def df_face(vid, num_frames, net):
    img, count = face_rec(extract_frames(vid, num_frames))
    return preprocess_frame(img) if count > 0 else []


def is_video(vid):
    return os.path.isfile(vid) and vid.lower().endswith(
        (".avi", ".mp4", ".mpg", ".mpeg", ".mov")
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(result, filename, y, y_val, klass, correct_label=None, compression=None):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))
    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)
    if compression is not None:
        result["video"]["compression"].append(compression)
    return result
