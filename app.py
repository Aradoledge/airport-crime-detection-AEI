import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
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

# Suppress WebRTC and asyncio logging to reduce error noise
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

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

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Live Monitoring")

    if not st.session_state.system_status["yolo_loaded"]:
        st.error("⚠️ YOLO detector failed to load. Basic object detection unavailable.")
    if not st.session_state.system_status["anomaly_model_loaded"]:
        st.warning("⚠️ Anomaly model not loaded. Anomaly detection will be limited.")

    # WebRTC streamer with empty configuration to avoid STUN errors
    webrtc_ctx = webrtc_streamer(
        key="airport-security",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({}),  # Empty config to prevent STUN errors
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        desired_playing_state=True,
        async_processing=True,
    )

    # Display current detection info below the video
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
        st.subheader("Current Frame Analysis")
        col1a, col2a, col3a = st.columns(3)

        with col1a:
            st.metric("Frame", frame_count)

        with col2a:
            st.metric("Anomaly Score", f"{anomaly_score:.3f}")

        with col3a:
            st.metric("Detections", len(detections))

        # Display detection details
        if detections:
            st.subheader("Detected Objects")
            for i, detection in enumerate(detections[:5]):  # Show first 5 detections
                st.write(
                    f"{i+1}. **{detection['class']}** (confidence: {detection['confidence']:.2f})"
                )
    else:
        st.session_state.system_status["camera_active"] = False
        st.info(
            "🔴 Camera feed not active. Please allow camera access and click 'START' to begin streaming."
        )

with col2:
    st.header("🚨 Alert Panel")

    # Display latest alert
    if st.session_state.alerts:
        latest_alert = list(st.session_state.alerts)[-1]  # Get last alert from deque
        st.markdown(
            f"""
        <div class="alert-box">
            <h3>🚨 SECURITY ALERT</h3>
            <p><strong>Time:</strong> {latest_alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Message:</strong> {latest_alert['message']}</p>
            <p><strong>Type:</strong> {latest_alert['type'].upper()}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ No active alerts")

    st.markdown("---")

    # System status
    st.subheader("📊 System Statistics")

    col11, col12, col13 = st.columns(3)

    with col11:
        st.metric("Frames Processed", st.session_state.detection_stats["total_frames"])

    with col12:
        st.metric(
            "Alerts Triggered", st.session_state.detection_stats["anomalies_detected"]
        )

    with col13:
        st.metric(
            "Objects Detected", st.session_state.detection_stats["objects_detected"]
        )

    st.markdown("---")

    # Detection statistics
    st.subheader("📈 Detection Statistics")

    # Create simple chart
    if st.session_state.detection_stats["total_frames"] > 0:
        fig = go.Figure()

        frames = st.session_state.detection_stats["total_frames"]
        anomalies = st.session_state.detection_stats["anomalies_detected"]
        objects = st.session_state.detection_stats["objects_detected"]

        fig.add_trace(
            go.Bar(
                x=["Frames", "Alerts", "Objects"],
                y=[frames, anomalies, objects],
                marker_color=["blue", "red", "green"],
            )
        )

        fig.update_layout(
            title="Detection Statistics",
            showlegend=False,
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

# Additional sections
st.markdown("---")

# Historical data and logs
col3, col4 = st.columns(2)

with col3:
    st.header("📋 Recent Alerts")

    if st.session_state.alerts:
        # Show last 5 alerts
        recent_alerts = list(st.session_state.alerts)[-5:]
        for alert in reversed(recent_alerts):
            alert_time = alert["timestamp"].strftime("%H:%M:%S")
            st.warning(f"**{alert_time}** - {alert['message']}")
    else:
        st.info("No alerts recorded yet")

with col4:
    st.header("⚙️ System Controls")

    col4a, col4b = st.columns(2)

    with col4a:
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.detection_stats = {
                "total_frames": 0,
                "anomalies_detected": 0,
                "objects_detected": 0,
                "last_update": time.time(),
            }
            st.session_state.alerts.clear()
            st.rerun()

    with col4b:
        if st.button("🗑️ Clear Alerts", use_container_width=True):
            st.session_state.alerts.clear()
            st.rerun()

    # Model information
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

# Auto-refresh for stats (only in main thread)
if webrtc_ctx.state.playing:
    st.session_state.system_status["last_health_check"] = time.time()
    time.sleep(0.1)
