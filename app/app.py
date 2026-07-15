"""
app.py — Helmet Detection App

Run with:
    streamlit run app/app.py

Requires trained weights at app/model/best.pt (see README for how to get these).
"""
import time
import io
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page config + dark theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Helmet Detection", page_icon="⛑️", layout="wide")

DARK_CSS = """
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stSidebar"] { background-color: #161a23; }
.metric-card {
    background-color: #1c212b; border-radius: 10px; padding: 16px;
    text-align: center; border: 1px solid #2a2f3a;
}
.metric-value { font-size: 28px; font-weight: 700; color: #4ade80; }
.metric-label { font-size: 13px; color: #9ca3af; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
CLASS_COLORS = {"helmet": (74, 222, 128), "no_helmet": (248, 113, 113), "person": (250, 204, 21)}


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def get_model():
    if not MODEL_PATH.exists():
        st.error(
            f"No model weights found at `{MODEL_PATH}`. Train a model (see README "
            f"Step 4) and copy your best.pt there before using this app."
        )
        st.stop()
    return load_model(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Session state for detection history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, source, counts, avg_conf, proc_time


def log_detection(source_type, counts, avg_conf, proc_time):
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "helmet": counts.get("helmet", 0),
        "no_helmet": counts.get("no_helmet", 0),
        "person": counts.get("person", 0),
        "total": sum(counts.values()),
        "avg_confidence": round(avg_conf, 3),
        "processing_time_ms": round(proc_time * 1000, 1),
    })


# ---------------------------------------------------------------------------
# Core detection + drawing helper
# ---------------------------------------------------------------------------
def run_detection(model, image_bgr, conf_threshold):
    start = time.time()
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)[0]
    proc_time = time.time() - start

    counts = {"helmet": 0, "no_helmet": 0, "person": 0}
    confidences = []
    annotated = image_bgr.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        counts[cls_name] = counts.get(cls_name, 0) + 1
        confidences.append(conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return annotated, counts, confidences, avg_conf, proc_time


def show_stats(counts, avg_conf, proc_time):
    c1, c2, c3, c4 = st.columns(4)
    total = sum(counts.values())
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Objects", "Avg Confidence", "Processing Time", "No-Helmet Count"],
        [total, f"{avg_conf:.2%}", f"{proc_time*1000:.0f} ms", counts.get("no_helmet", 0)],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.write(f"Breakdown — helmet: **{counts.get('helmet',0)}**, "
             f"no_helmet: **{counts.get('no_helmet',0)}**, "
             f"person: **{counts.get('person',0)}**")


def download_button_for_image(annotated_bgr, filename):
    rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    st.download_button("Export annotated image", data=buf.getvalue(),
                        file_name=filename, mime="image/png")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⛑️ Helmet Detection")
mode = st.sidebar.radio(
    "Mode",
    ["Image Upload", "Video Upload", "Webcam (Live)", "Dashboard", "Detection History"],
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.caption("Classes: helmet · no_helmet · person")

model = get_model() if mode not in ("Dashboard", "Detection History") else None

# ---------------------------------------------------------------------------
# Image Upload
# ---------------------------------------------------------------------------
if mode == "Image Upload":
    st.header("Image Upload Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, counts, confidences, avg_conf, proc_time = run_detection(
            model, image_bgr, conf_threshold
        )

        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

        show_stats(counts, avg_conf, proc_time)
        download_button_for_image(annotated, f"detected_{uploaded.name}")
        log_detection("image", counts, avg_conf, proc_time)

# ---------------------------------------------------------------------------
# Video Upload
# ---------------------------------------------------------------------------
elif mode == "Video Upload":
    st.header("Video Upload Detection")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp_path = Path("temp_uploaded_video.mp4")
        tmp_path.write_bytes(uploaded.read())

        cap = cv2.VideoCapture(str(tmp_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = Path("temp_annotated_video.mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_placeholder = st.empty()
        progress = st.progress(0)
        stat_placeholder = st.empty()

        all_counts = {"helmet": 0, "no_helmet": 0, "person": 0}
        all_confidences = []
        total_proc_time = 0.0
        frame_idx = 0

        run_it = st.button("Run detection on this video")
        if run_it:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, counts, confidences, avg_conf, proc_time = run_detection(
                    model, frame, conf_threshold
                )
                for k, v in counts.items():
                    all_counts[k] += v
                all_confidences.extend(confidences)
                total_proc_time += proc_time
                writer.write(annotated)

                frame_idx += 1
                if frame_idx % 5 == 0:  # throttle UI updates
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}/{total_frames}",
                        use_container_width=True,
                    )
                    progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

            cap.release()
            writer.release()

            overall_avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
            with stat_placeholder.container():
                show_stats(all_counts, overall_avg_conf, total_proc_time)
                st.caption(f"Processed {frame_idx} frames in {total_proc_time:.1f}s "
                           f"({frame_idx/max(total_proc_time,1e-6):.1f} FPS)")

            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button("Export annotated video", data=f.read(),
                                    file_name=f"detected_{uploaded.name}", mime="video/mp4")
            log_detection("video", all_counts, overall_avg_conf, total_proc_time)

# ---------------------------------------------------------------------------
# Webcam (Live) — using streamlit-webrtc
# ---------------------------------------------------------------------------
elif mode == "Webcam (Live)":
    st.header("Live Webcam Detection")
    st.caption("Grant camera permission in your browser when prompted.")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class HelmetDetectionProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = get_model()
            self.conf = conf_threshold
            self.last_counts = {"helmet": 0, "no_helmet": 0, "person": 0}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, counts, confidences, avg_conf, proc_time = run_detection(
                self.model, img, self.conf
            )
            self.last_counts = counts
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="helmet-webcam",
        video_processor_factory=HelmetDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.conf = conf_threshold
        if st.button("Log current frame stats to history"):
            counts = ctx.video_processor.last_counts
            log_detection("webcam", counts, 0.0, 0.0)
            st.success("Logged.")

# ---------------------------------------------------------------------------
# Dashboard (bonus)
# ---------------------------------------------------------------------------
elif mode == "Dashboard":
    st.header("Detection Dashboard")
    if not st.session_state.history:
        st.info("No detections logged yet. Run some detections first.")
    else:
        df = pd.DataFrame(st.session_state.history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", len(df))
        c2.metric("Total Helmets", int(df["helmet"].sum()))
        c3.metric("Total No-Helmet", int(df["no_helmet"].sum()))
        c4.metric("Avg Processing Time (ms)", f"{df['processing_time_ms'].mean():.1f}")

        st.subheader("Compliance over time")
        st.bar_chart(df.set_index("timestamp")[["helmet", "no_helmet"]])

        st.subheader("Detections by source type")
        st.bar_chart(df.groupby("source")["total"].sum())

        compliance_rate = (
            df["helmet"].sum() / max(df["helmet"].sum() + df["no_helmet"].sum(), 1)
        )
        st.metric("Overall Helmet Compliance Rate", f"{compliance_rate:.1%}")

# ---------------------------------------------------------------------------
# Detection History / Logs (bonus)
# ---------------------------------------------------------------------------
elif mode == "Detection History":
    st.header("Detection History / Logs")
    if not st.session_state.history:
        st.info("No detections logged yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export logs as CSV", data=csv, file_name="detection_logs.csv",
                            mime="text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
