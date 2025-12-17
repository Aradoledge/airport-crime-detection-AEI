"""
Reliable Streamlit-WebRTC Object Detection Demo
Based on the official MobileNet SSD example
This version is STABLE on Streamlit Cloud
"""

import os
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_session_memo import st_session_memo
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc

# ------------------------------------------------------------------
# Environment hardening (SAFE)
# ------------------------------------------------------------------
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Reliable WebRTC Object Detection",
    layout="wide",
)

st.title("📹 Stable WebRTC Object Detection Demo")

# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------
HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_URL = (
    "https://github.com/robmarkcole/object-detection-app/raw/master/"
    "model/MobileNetSSD_deploy.caffemodel"
)
PROTOTXT_URL = (
    "https://github.com/robmarkcole/object-detection-app/raw/master/"
    "model/MobileNetSSD_deploy.prototxt.txt"
)

MODEL_PATH = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH = MODELS_DIR / "MobileNetSSD_deploy.prototxt.txt"

# ------------------------------------------------------------------
# Download helper
# ------------------------------------------------------------------
def download(url: str, dst: Path):
    if dst.exists():
        return
    import requests

    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

download(MODEL_URL, MODEL_PATH)
download(PROTOTXT_URL, PROTOTXT_PATH)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

@st.cache_resource
def generate_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_colors()

@st_session_memo
def load_model():
    return cv2.dnn.readNetFromCaffe(
        str(PROTOTXT_PATH),
        str(MODEL_PATH),
    )

net = load_model()

# ------------------------------------------------------------------
# UI controls
# ------------------------------------------------------------------
score_threshold = st.slider(
    "Detection confidence threshold",
    0.0, 1.0, 0.5, 0.05
)

# Thread-safe communication queue
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# ------------------------------------------------------------------
# WebRTC video callback (NO STREAMLIT CALLS HERE)
# ------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
    )

    net.setInput(blob)
    output = net.forward().squeeze()

    detections: List[Detection] = []

    for det in output:
        confidence = float(det[2])
        if confidence < score_threshold:
            continue

        class_id = int(det[1])
        box = det[3:7] * np.array([w, h, w, h])

        detections.append(
            Detection(
                class_id=class_id,
                label=CLASSES[class_id],
                score=confidence,
                box=box,
            )
        )

        xmin, ymin, xmax, ymax = box.astype(int)
        color = COLORS[class_id]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            f"{CLASSES[class_id]} {confidence:.2f}",
            (xmin, max(ymin - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# ------------------------------------------------------------------
# WebRTC streamer (CORRECT CONFIG)
# ------------------------------------------------------------------
webrtc_ctx = webrtc_streamer(
    key="stable-object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ------------------------------------------------------------------
# UI thread (SAFE)
# ------------------------------------------------------------------
st.markdown("### 📊 Detection Results")

if webrtc_ctx.state.playing:
    table_placeholder = st.empty()

    while webrtc_ctx.state.playing:
        detections = result_queue.get()
        table_placeholder.table(detections)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
**Versions**
- Streamlit: `{st.__version__}`
- Streamlit-WebRTC: `{st_webrtc_version}`
- aiortc: `{aiortc.__version__}`
"""
)
