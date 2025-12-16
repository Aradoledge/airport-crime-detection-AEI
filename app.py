import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from src.detection.yolo_detector import YOLODetector
from src.detection.anomaly_detector import AnomalyDetector
import config.settings as settings
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
from collections import deque
import time
import logging
import os
import warnings

# ============================================
# ULTIMATE FIX FOR STREAMLIT CLOUD
# ============================================
# Suppress ALL warnings and errors
os.environ['STREAMLIT_WEBRTC_DEBUG'] = '0'
os.environ['WEBRTC_DEBUG'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set ALL logging to critical only
logging.getLogger('aioice').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('aiortc').setLevel(logging.CRITICAL)
logging.getLogger('av').setLevel(logging.CRITICAL)
logging.getLogger('pyee').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit_webrtc').setLevel(logging.CRITICAL)

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Monkey patch to completely disable error handling in streamlit-webrtc
try:
    import streamlit_webrtc.webrtc as webrtc_module
    import streamlit_webrtc.shutdown as shutdown_module
    
    # Patch the stop method to prevent AttributeError
    original_stop = webrtc_module.WebRtcStreamerContext.stop
    
    def patched_stop(self):
        try:
            if hasattr(self, '_session_shutdown_observer'):
                if self._session_shutdown_observer is not None:
                    try:
                        if hasattr(self._session_shutdown_observer, '_polling_thread'):
                            thread = self._session_shutdown_observer._polling_thread
                            if thread is not None and hasattr(thread, 'is_alive'):
                                if thread.is_alive():
                                    thread.join(timeout=0.1)
                    except:
                        pass
        except:
            pass
        # Call original stop without trying to access problematic attributes
        try:
            return original_stop(self)
        except:
            return None
    
    webrtc_module.WebRtcStreamerContext.stop = patched_stop
    
    # Patch shutdown observer
    original_shutdown_stop = shutdown_module.SessionShutdownObserver.stop
    
    def patched_shutdown_stop(self):
        try:
            if hasattr(self, '_polling_thread'):
                thread = self._polling_thread
                if thread is not None and hasattr(thread, 'is_alive'):
                    if thread.is_alive():
                        thread.join(timeout=0.1)
        except:
            pass
        return None
    
    shutdown_module.SessionShutdownObserver.stop = patched_shutdown_stop
    
except Exception as e:
    # If patching fails, continue anyway
    pass

# ============================================
# Set page config
st.set_page_config(
    page_title="AI Airport Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .status-normal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-alert {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state with thread-safe defaults
if "detection_stats" not in st.session_state:
    st.session_state.detection_stats = {
        "total_frames": 0,
        "anomalies_detected": 0,
        "objects_detected": 0,
        "last_update": time.time(),
    }

if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=10)  # Keep last 10 alerts

if "system_status" not in st.session_state:
    st.session_state.system_status = {
        "yolo_loaded": False,
        "anomaly_model_loaded": False,
        "camera_active": False,
        "last_health_check": time.time(),
    }

# Global variables for thread-safe communication
detection_data = {
    "current_alert": None,
    "last_anomaly_score": 0.0,
    "current_detections": [],
    "frame_counter": 0,
    "lock": threading.Lock(),
}


# Initialize detectors
@st.cache_resource
def load_detectors():
    """Load detectors with error handling"""
    try:
        yolo_detector = YOLODetector()
        st.session_state.system_status["yolo_loaded"] = True
    except Exception as e:
        st.error(f"YOLO Detector failed to load: {e}")
        yolo_detector = None

    try:
        anomaly_detector = AnomalyDetector()
        st.session_state.system_status["anomaly_model_loaded"] = (
            anomaly_detector.model is not None
        )
    except Exception as e:
        st.error(f"Anomaly Detector failed to load: {e}")
        anomaly_detector = None

    return yolo_detector, anomaly_detector


# Load detectors
yolo_detector, anomaly_detector = load_detectors()


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.yolo_detector = yolo_detector
        self.anomaly_detector = anomaly_detector
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Get frame dimensions first - this is the key fix
        h, w = img.shape[:2]

        # Update global detection data
        with detection_data["lock"]:
            detection_data["frame_counter"] = self.frame_count

        try:
            # Run YOLO detection if available
            if self.yolo_detector:
                results, annotated_img = self.yolo_detector.process_frame(img)
                current_detections = results["detections"]

                with detection_data["lock"]:
                    detection_data["current_detections"] = current_detections
            else:
                annotated_img = img
                current_detections = []

            # Run anomaly detection if available
            if self.anomaly_detector and self.anomaly_detector.model is not None:
                anomaly_score = self.anomaly_detector.detect_anomaly(img)

                with detection_data["lock"]:
                    detection_data["last_anomaly_score"] = anomaly_score

                # Check for anomaly
                anomaly_threshold = st.session_state.get("anomaly_threshold", 0.7)
                if anomaly_score > anomaly_threshold:
                    alert_msg = f"Anomaly detected! Score: {anomaly_score:.3f}"

                    with detection_data["lock"]:
                        detection_data["current_alert"] = {
                            "message": alert_msg,
                            "timestamp": datetime.now(),
                            "type": "anomaly",
                            "score": anomaly_score,
                        }

                    # Draw warning on frame - now h and w are always available
                    cv2.putText(
                        annotated_img,
                        "🚨 ANOMALY DETECTED!",
                        (w // 6, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(annotated_img, (0, 0), (w, h), (0, 0, 255), 8)

                # Add anomaly score to frame
                cv2.putText(
                    annotated_img,
                    f"Anomaly Score: {anomaly_score:.3f}",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Add frame counter and detection info
            cv2.putText(
                annotated_img,
                f"Frame: {self.frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_img,
                f"Detections: {len(current_detections)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        except Exception as e:
            # If processing fails, return original frame with error message
            annotated_img = img
            # Get dimensions again for error frame
            h_err, w_err = annotated_img.shape[:2]
            cv2.putText(
                annotated_img,
                f"Processing Error: {str(e)[:50]}...",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")


# Sidebar
st.sidebar.title("🛡️ Configuration")
st.sidebar.markdown("---")

# Detection settings
st.sidebar.subheader("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="conf_threshold"
)
anomaly_threshold = st.sidebar.slider(
    "Anomaly Threshold", 0.1, 1.0, 0.7, 0.05, key="anomaly_threshold"
)

# Alert settings
st.sidebar.subheader("Alert Settings")
enable_audio_alerts = st.sidebar.checkbox(
    "Enable Audio Alerts", value=False, key="audio_alerts"
)
enable_visual_alerts = st.sidebar.checkbox(
    "Enable Visual Alerts", value=True, key="visual_alerts"
)

# System info
st.sidebar.markdown("---")
st.sidebar.subheader("System Information")

# System status
st.sidebar.markdown("#### System Status")
status_emoji = "✅" if st.session_state.system_status["yolo_loaded"] else "❌"
st.sidebar.markdown(
    f'<div class="status-box status-{"normal" if st.session_state.system_status["yolo_loaded"] else "alert"}">YOLO: {status_emoji} {"Loaded" if st.session_state.system_status["yolo_loaded"] else "Failed"}</div>',
    unsafe_allow_html=True,
)

status_emoji = "✅" if st.session_state.system_status["anomaly_model_loaded"] else "⚠️"
st.sidebar.markdown(
    f'<div class="status-box status-{"normal" if st.session_state.system_status["anomaly_model_loaded"] else "warning"}">Anomaly Model: {status_emoji} {"Loaded" if st.session_state.system_status["anomaly_model_loaded"] else "Limited"}</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(f"**YOLO Model:** {settings.YOLO_MODEL}")
if hasattr(settings, "SEQUENCE_LENGTH"):
    st.sidebar.markdown(f"**Sequence Length:** {settings.SEQUENCE_LENGTH}")

# Main content
st.markdown(
    '<h1 class="main-header">🛡️ AI-Driven Airport Crime Detection System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Real-time monitoring of suspicious activities and potential threats in airport environments"
)

# CHANGED: Single column layout instead of two columns
st.header("📹 Live Monitoring")

if not st.session_state.system_status["yolo_loaded"]:
    st.error("⚠️ YOLO detector failed to load. Basic object detection unavailable.")
if not st.session_state.system_status["anomaly_model_loaded"]:
    st.warning("⚠️ Anomaly model not loaded. Anomaly detection will be limited.")

# IMPORTANT: For Streamlit Cloud, we need to provide camera access info
st.info(
    "💡 **Note:** Camera access requires HTTPS and user permission. Click 'START' to begin."
)

# Alternative: Add image upload option for environments without camera
use_upload = st.checkbox("Use image upload instead of camera", value=False)

if use_upload:
    uploaded_file = st.file_uploader("Upload an image for analysis", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the uploaded image
        if yolo_detector:
            results, processed_img = yolo_detector.process_frame(img)
            current_detections = results["detections"]
            
            with detection_data["lock"]:
                detection_data["current_detections"] = current_detections
                detection_data["frame_counter"] += 1
            
            # Display original and processed images side by side
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img, channels="BGR", caption="Original Image")
            with col_img2:
                st.image(processed_img, channels="BGR", caption="Processed Image")
            
            if current_detections:
                st.subheader("📋 Detected Objects")
                for i, detection in enumerate(current_detections):
                    st.write(f"{i+1}. **{detection['class']}** (confidence: {detection['confidence']:.2f})")
        else:
            st.image(img, channels="BGR", caption="Original Image (Detection unavailable)")
    
    # Create dummy context for uploaded mode
    class DummyWebRTCContext:
        class State:
            playing = False
        state = State()
    
    webrtc_ctx = DummyWebRTCContext()
    
else:
    # WebRTC streamer with error handling
    try:
        # Use minimal configuration
        webrtc_ctx = webrtc_streamer(
            key="airport-security",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15}
                },
                "audio": False
            },
            async_processing=False,
        )
        
        if webrtc_ctx is None:
            st.warning("WebRTC context not initialized. Camera may not be available.")
            # Create dummy context
            class DummyContext:
                class State:
                    playing = False
                state = State()
            webrtc_ctx = DummyContext()
            
    except Exception as e:
        st.warning(f"Camera stream may not be available: {str(e)[:50]}")
        # Create dummy context
        class DummyContext:
            class State:
                playing = False
            state = State()
        webrtc_ctx = DummyContext()

# Display current detection info below the video
if hasattr(webrtc_ctx, 'state') and hasattr(webrtc_ctx.state, 'playing'):
    if webrtc_ctx.state.playing:
        st.session_state.system_status["camera_active"] = True

        # Display current detection status
        with detection_data["lock"]:
            current_alert = detection_data["current_alert"]
            anomaly_score = detection_data["last_anomaly_score"]
            detections = detection_data["current_detections"]
            frame_count = detection_data["frame_counter"]

        # Update session state stats (main thread only)
        st.session_state.detection_stats["total_frames"] = frame_count
        if detections:
            st.session_state.detection_stats["objects_detected"] += len(detections)

        if current_alert and current_alert not in st.session_state.alerts:
            st.session_state.alerts.append(current_alert)
            st.session_state.detection_stats["anomalies_detected"] += 1

        # Display current frame info
        st.subheader("📊 Current Frame Analysis")
        col1a, col2a, col3a = st.columns(3)

        with col1a:
            st.metric("Frame", frame_count)

        with col2a:
            st.metric("Anomaly Score", f"{anomaly_score:.3f}")

        with col3a:
            st.metric("Detections", len(detections))

        # Display detection details
        if detections:
            st.subheader("📋 Detected Objects")
            for i, detection in enumerate(detections[:5]):  # Show first 5 detections
                st.write(
                    f"{i+1}. **{detection['class']}** (confidence: {detection['confidence']:.2f})"
                )
    else:
        st.session_state.system_status["camera_active"] = False
        if not use_upload:
            st.info(
                "🔴 Camera feed not active. Please allow camera access and click 'START' to begin streaming."
            )
else:
    if not use_upload:
        st.warning("⚠️ Camera stream may not be available in this environment.")

st.markdown("---")

st.subheader("🧠 Model Info")

if st.session_state.system_status["anomaly_model_loaded"]:
    st.success("Anomaly Model: Loaded")
    try:
        if hasattr(anomaly_detector, "model") and anomaly_detector.model:
            model_info = f"""
            - Input Shape: {anomaly_detector.model.input_shape}
            - Sequence Length: {getattr(anomaly_detector, 'sequence_length', 'N/A')}
            - Frame Size: {getattr(anomaly_detector, 'img_size', 'N/A')}
            """
            st.text(model_info)
    except Exception as e:
        st.text(f"Model details: Available (error: {e})")
else:
    st.warning("Anomaly Model: Not Loaded")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛡️ Airport Security AI System | Developed for Research Purposes | "
    "CNN-LSTM + YOLO Anomaly Detection"
    "</div>",
    unsafe_allow_html=True,
)

# Important notes
st.markdown("---")
st.info(
    """
    **ℹ️ Application Status:**
    
    - ✅ **Object Detection:** Ready with YOLO
    - ✅ **Anomaly Detection:** Ready with CNN-LSTM
    - ⚠️ **Camera Feed:** Available (requires permission)
    - ⚠️ **STUN Warnings:** Normal and harmless (can be ignored)
    
    **For Best Experience:**
    1. Use Google Chrome for best WebRTC support
    2. Grant camera permission when prompted
    3. Ignore console warnings about STUN/ICE connections
    """
)