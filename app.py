"""Streamlit operator console for the Real-Time Industrial PPE Safety Detection System."""

from __future__ import annotations

import time
from datetime import datetime, time as clock_time, timezone
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.database import ViolationDatabase
from src.detector import PPEDetector
from src.utils import resize_frame, save_snapshot

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
DATABASE = ROOT / "violations.db"

st.set_page_config(page_title="PPE Safety Detection", page_icon="⛑️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_database() -> ViolationDatabase:
    """Reuse one database manager across Streamlit reruns."""
    return ViolationDatabase(DATABASE)


@st.cache_resource(show_spinner="Loading PPE model…")
def get_detector(model_file: str, confidence: float) -> PPEDetector:
    """Cache model initialisation; changing threshold creates a fresh detector."""
    return PPEDetector(model_file, confidence=confidence)


def resolve_model() -> Path | None:
    """Return the first local PPE PT/ONNX model, prioritising ONNX edge models."""
    candidates = sorted(MODELS.glob("*.onnx")) + sorted(MODELS.glob("*.pt"))
    return candidates[0] if candidates else None


def live_feed() -> None:
    """Render webcam controls and process frames until the user stops the feed."""
    st.sidebar.header("Detection controls")
    threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.05)
    snapshot_cooldown = st.sidebar.slider("Snapshot cooldown (seconds)", 1, 60, 10)
    model = resolve_model()
    st.sidebar.caption(f"Model: `{model.name}`" if model else "No model found in `models/`.")
    if not model:
        st.info("Place your PPE-trained `.pt` or `.onnx` weights in `models/`, then refresh this page. Standard COCO weights cannot identify PPE compliance.")
        return
    try:
        detector = get_detector(str(model), threshold)
    except Exception as error:
        st.error(f"Model unavailable: {error}")
        return
    options = ["All", *sorted(set(detector.names.values()))]
    target = st.sidebar.selectbox("Target class", options)
    if "running" not in st.session_state:
        st.session_state.running = False
    start, stop = st.columns(2)
    if start.button("Start webcam", type="primary", use_container_width=True):
        st.session_state.running = True
    if stop.button("Stop", use_container_width=True):
        st.session_state.running = False
    video = st.empty()
    metrics = st.columns(4)
    last_logged = float(st.session_state.get("last_logged", 0))
    if not st.session_state.running:
        st.caption("Connect a camera, then select **Start webcam**. The feed runs locally in this process.")
        return
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        st.session_state.running = False
        st.error("Camera 0 could not be opened. Check camera permissions or close other camera applications.")
        return
    database = get_database()
    try:
        while st.session_state.running:
            ok, frame = camera.read()
            if not ok:
                st.warning("Camera frame could not be read; stopping feed.")
                break
            annotated, counts, detections, fps = detector.process_frame(resize_frame(frame), target)
            video.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            metrics[0].metric("FPS", f"{fps:.1f}")
            metrics[1].metric("People", counts["people"])
            metrics[2].metric("PPE detected", counts["compliant"])
            metrics[3].metric("Violations", counts["violations"])
            violations = [item for item in detections if item.violation]
            now = time.time()
            if violations and now - last_logged >= snapshot_cooldown:
                path = save_snapshot(annotated, LOGS)
                for item in violations:
                    database.log_violation_async(item.label, item.confidence, path)
                st.session_state.last_logged = now
                last_logged = now
                st.error("PPE violation detected and recorded in the audit log.")
    except Exception as error:
        st.error(f"Live processing stopped: {error}")
    finally:
        camera.release()
        st.session_state.running = False


def analytics() -> None:
    """Render audit records with UTC date filtering and snapshot preview."""
    st.subheader("Violation audit trail")
    left, right = st.columns(2)
    with left:
        start_date = st.date_input("From date", value=None)
    with right:
        end_date = st.date_input("To date", value=None)
    start = datetime.combine(start_date, clock_time.min, tzinfo=timezone.utc) if start_date else None
    end = datetime.combine(end_date, clock_time.max, tzinfo=timezone.utc) if end_date else None
    if start and end and start > end:
        st.error("The start date must be on or before the end date.")
        return
    records = get_database().query(start, end)
    if records.empty:
        st.info("No violations have been recorded for this window.")
        return
    st.dataframe(records, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", records.to_csv(index=False).encode("utf-8"), "ppe_violations.csv", "text/csv")
    choices = records["id"].astype(str).tolist()
    selected_id = st.selectbox("Preview violation snapshot", choices)
    selected = records.loc[records["id"].astype(str) == selected_id].iloc[0]
    image_path = Path(selected["snapshot_path"])
    if image_path.is_file():
        st.image(str(image_path), caption=f"{selected['violation_type']} · {selected['timestamp']}", use_container_width=True)
    else:
        st.warning("The snapshot file is no longer available on disk.")


st.title("⛑️ Real-Time Industrial PPE Safety Detection")
st.caption("Local edge-ready computer vision with auditable violation logging.")
feed_tab, analytics_tab = st.tabs(["Live Feed", "Analytics Dashboard"])
with feed_tab:
    live_feed()
with analytics_tab:
    analytics()
