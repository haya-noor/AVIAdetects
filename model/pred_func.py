import os
import numpy as np
import cv2
import torch
import dlib
# import face_recognition
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    model = GenConViT(
        config,
        ed= ed_weight,
        vae= vae_weight, 
        net=net,
        fp16=fp16
    )

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


# ──────────────────────────────  FACE EXTRACTION  ─────────────────────────────
def face_rec(frames, p=None, klass=None):
    """
    Extract one 224×224 RGB crop per video frame using dlib’s HOG or CUDA CNN
    detector.  Returns (faces_array, count) just like the original function.

    frames : np.ndarray[N, H, W, 3]  – RGB frames from decord
    """
    # Pre-allocate output (same logic as before)
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0

    # Pick fastest detector available
    detector = (
        dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        if dlib.DLIB_USE_CUDA
        else dlib.get_frontal_face_detector()
    )

    for frame in frames:
        # dlib expects RGB
        rgb = frame if frame.shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect
        dets = detector(rgb, 0)  # 0 = no upsampling

        # cnn_face_detection_model_v1 returns objects with .rect; HOG returns rectangles
        rectangles = [d.rect if hasattr(d, "rect") else d for d in dets]

        for rect in rectangles:
            if count >= len(frames):
                break

            top, bottom = max(0, rect.top()), min(rgb.shape[0], rect.bottom())
            left, right = max(0, rect.left()), min(rgb.shape[1], rect.right())

            if bottom - top == 0 or right - left == 0:
                continue  # skip degenerate boxes

            face_img = rgb[top:bottom, left:right]
            face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
            temp_face[count] = face_img
            count += 1

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
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)  # Calculate the step size between frames
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size


def df_face(vid, num_frames, net):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    return preprocess_frame(face) if count > 0 else []


def is_video(vid):
    print('IS FILE', os.path.isfile(vid))
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
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


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
