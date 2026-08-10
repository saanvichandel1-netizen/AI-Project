"""Professional Streamlit dashboard for the HelpGuard AI safety monitor."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

from .detection import SafetyStateProcessor
from .incidents import append_incident, ensure_log_file


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "test_video.mp4"


def _inject_styles() -> None:
    """Apply a polished visual theme for the dashboard."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .stMetric { background: linear-gradient(90deg, #0f172a 0%, #111827 100%); border: 1px solid #334155; border-radius: 14px; padding: 0.7rem 0.8rem; }
        .stButton>button { border-radius: 999px; }
        .main-header { background: linear-gradient(90deg, #0f172a, #1f2937); border-radius: 18px; padding: 1.2rem 1.3rem; margin-bottom: 1rem; }
        .subtle { color: #94a3b8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> str:
    """Return a simple HTML badge for the current monitoring status."""
    if status == "Alert":
        return '<span style="color:#f43f5e; font-weight:700;">⚠️ Alert</span>'
    return '<span style="color:#22c55e; font-weight:700;">● Monitoring</span>'


def _load_video_frames(path: Path, max_frames: int) -> List[np.ndarray]:
    """Load a bounded number of frames from a local video file."""
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to read video from {path}")

    try:
        while len(frames) < max_frames:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _clear_session_state() -> None:
    """Clear the current analysis state from Streamlit session memory."""
    for key in ["latest_metrics", "latest_frame", "analysis_summary", "incident_rows"]:
        st.session_state.pop(key, None)


def run_app() -> None:
    """Render the HelpGuard AI dashboard and run the monitoring workflow."""
    _inject_styles()
    st.set_page_config(page_title="HelpGuard AI", page_icon="🛡️", layout="wide")

    st.markdown(
        """
        <div class="main-header">
            <h1 style='margin:0; color:#f8fafc;'>🛡️ HelpGuard AI</h1>
            <p style='margin:0.25rem 0 0 0; color:#cbd5e1;'>Explainable computer-vision monitoring for fatigue-related facial cues</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Built locally with Python, Streamlit, OpenCV, MediaPipe, and NumPy for a practical academic demo.")

    processor = SafetyStateProcessor()
    selected_video = DEFAULT_VIDEO

    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        video_source = st.file_uploader("Upload a local video", type=["mp4", "mov", "avi"])
        if video_source is not None:
            temp_path = ROOT / video_source.name
            temp_path.write_bytes(video_source.getvalue())
            selected_video = temp_path

        st.caption(f"Source: {selected_video.name}")
        ear_threshold = st.slider("EAR threshold", 0.15, 0.35, 0.25, step=0.01)
        mar_threshold = st.slider("MAR threshold", 0.30, 0.70, 0.45, step=0.01)
        max_frames = st.slider("Frames to analyze", 10, 120, 45, step=5)

        col_a, col_b = st.columns(2)
        with col_a:
            run_analysis = st.button("▶ Run analysis", use_container_width=True)
        with col_b:
            reset_button = st.button("↺ Reset", use_container_width=True)

        st.markdown("---")
        st.info("The system uses MediaPipe Face Mesh to estimate EAR and MAR and to flag fatigue-related facial cues.")
        st.caption("Tip: use a short video with clear facial visibility for the best results.")

    if reset_button:
        _clear_session_state()
        st.success("Dashboard reset. Upload a new video or run the sample again.")

    if not selected_video.exists():
        st.error("No valid video file is available. Upload a video or keep the sample file in the project folder.")
        return

    if run_analysis:
        with st.spinner("Analyzing video frames..."):
            try:
                frames = _load_video_frames(selected_video, max_frames)
            except Exception as exc:  # pragma: no cover - defensive UI handling
                st.error(f"Unable to read the selected video: {exc}")
                return

            progress_bar = st.progress(0)
            progress_text = st.empty()
            video_placeholder = st.empty()
            results = []
            start_time = time.perf_counter()

            for index, frame in enumerate(frames, start=1):
                result = processor.process(frame, ear_threshold=ear_threshold, mar_threshold=mar_threshold)
                results.append(result)

                st.session_state["latest_metrics"] = {
                    "ear": result["ear"],
                    "mar": result["mar"],
                    "risk_score": result["risk_score"],
                    "status": result["status"],
                    "reason": result.get("reason", "Normal monitoring."),
                }
                st.session_state["latest_frame"] = result["frame"]
                video_placeholder.image(
                    cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                percent = int((index / len(frames)) * 100)
                progress_bar.progress(index / len(frames))
                progress_text.write(f"Processing frame {index} of {len(frames)} ({percent}%)")

            elapsed_time = time.perf_counter() - start_time
            processed_frames = len(results)
            fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

            if not results:
                st.error("The video did not produce any frames for analysis.")
                return

            latest = results[-1]
            latest_metrics = {
                "ear": latest["ear"],
                "mar": latest["mar"],
                "risk_score": latest["risk_score"],
                "status": latest["status"],
                "reason": latest.get("reason", "Normal monitoring."),
            }
            alert_count = sum(1 for item in results if item["status"] == "Alert")
            average_ear = round(sum(item["ear"] for item in results) / processed_frames, 3)
            average_mar = round(sum(item["mar"] for item in results) / processed_frames, 3)
            max_risk = round(max(item["risk_score"] for item in results), 3)

            incident_rows = []
            for item in results:
                if item["status"] == "Alert":
                    incident_rows.append(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ear": round(item["ear"], 3),
                            "mar": round(item["mar"], 3),
                            "risk_score": round(item["risk_score"], 3),
                            "status": item["status"],
                        }
                    )
                    append_incident(incident_rows[-1])

            st.session_state["latest_metrics"] = latest_metrics
            st.session_state["latest_frame"] = latest["frame"]
            st.session_state["analysis_summary"] = {
                "alert_count": alert_count,
                "average_ear": average_ear,
                "average_mar": average_mar,
                "max_risk": max_risk,
                "processed_frames": processed_frames,
                "processing_time_seconds": round(elapsed_time, 2),
                "fps": round(fps, 2),
            }
            st.session_state["incident_rows"] = incident_rows

            st.success(
                f"Analysis completed for {processed_frames} frames in {elapsed_time:.2f}s ({fps:.1f} FPS)."
            )

    latest_frame = st.session_state.get("latest_frame")
    latest_metrics = st.session_state.get("latest_metrics")
    analysis_summary = st.session_state.get("analysis_summary")

    if latest_frame is None or latest_metrics is None:
        st.info("Select a video and press Run analysis to begin monitoring.")
        return

    col_left, col_right = st.columns([1.3, 0.7])
    with col_left:
        st.subheader("🖼️ Annotated frame")
        st.image(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    with col_right:
        st.subheader("📈 Monitoring overview")
        st.markdown(f"**Status:** {_status_badge(latest_metrics['status'])}")
        st.metric("Latest EAR", f"{latest_metrics['ear']:.3f}")
        st.metric("Latest MAR", f"{latest_metrics['mar']:.3f}")
        st.metric("Risk score", f"{latest_metrics['risk_score']:.3f}")

        if analysis_summary is not None:
            st.markdown("### Summary statistics")
            stat_cols = st.columns(2)
            stat_cols[0].metric("Alert count", analysis_summary["alert_count"])
            stat_cols[1].metric("Max risk", f"{analysis_summary['max_risk']:.3f}")
            st.metric("Average EAR", f"{analysis_summary['average_ear']:.3f}")
            st.metric("Average MAR", f"{analysis_summary['average_mar']:.3f}")
            st.metric("Frames analyzed", analysis_summary["processed_frames"])
            st.metric("Processing time", f"{analysis_summary['processing_time_seconds']:.2f}s")
            st.metric("Approx FPS", f"{analysis_summary['fps']:.2f}")

        st.markdown("### Why the system raised this result")
        st.info(latest_metrics["reason"])

    st.markdown("---")
    st.subheader("🧠 Explainability")
    with st.expander("What do EAR and MAR mean?"):
        st.write(
            "- EAR (Eye Aspect Ratio) measures how open the eyes appear. Lower EAR values often suggest the eyes are closing.\n"
            "- MAR (Mouth Aspect Ratio) measures mouth openness. Higher MAR values can indicate yawning.\n"
            "- The fatigue score combines both metrics into a simple, explainable risk estimate."
        )

    with st.expander("How the fatigue score is calculated"):
        st.write(
            "The software compares the current EAR and MAR values with the selected thresholds. If the eyes stay below the EAR threshold or the mouth rises above the MAR threshold, the system raises an alert."
        )

    st.markdown("---")
    st.subheader("📄 Incident log")
    if st.session_state.get("incident_rows"):
        st.caption("Alerts detected during the current run are saved locally as a CSV file.")
    else:
        st.caption("No alerts were recorded in the current run.")

    log_file = ensure_log_file()
    st.download_button(
        label="⬇️ Download incident CSV",
        data=log_file.read_bytes(),
        file_name="helpguard_incidents.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("HelpGuard AI • Built locally with Python, Streamlit, OpenCV, MediaPipe, and NumPy")
