import os
import cv2
import streamlit as st

from app.detection import SafetyStateProcessor

def configure_page():
    st.set_page_config(
        page_title="HelpGuard AI: Explainable Surveillance",
        page_icon="https://cdn-icons-png.flaticon.com/512/1828/1828843.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "HelpGuard AI monitors driver alertness and fatigue in real time.",
        },
    )

def run_app():
    configure_page()

    col1, col2 = st.columns(spec=[6, 2], gap="medium")

    with col1:
        st.title("HelpGuard AI: Driver Safety Monitor 🥱😪😴")
        with st.container():
            c1, c2 = st.columns(spec=[1, 1])
            with c1:
                WAIT_TIME = st.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)
            with c2:
                EAR_THRESH = st.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)

    thresholds = {
        "EAR_THRESH": EAR_THRESH,
        "WAIT_TIME": WAIT_TIME,
    }

    with col1:
        video_path = "test_video.mp4"
        st.subheader("Local video playback")

        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Unable to open the local video file.")
            else:
                video_handler = SafetyStateProcessor()
                placeholder = st.empty()
                
                with col2:
                    st.markdown("### Live EAR Data")
                    st.caption("Realtime driver alertness telemetry")
                    chart_placeholder = st.line_chart([])
                    ear_data = []

                    st.markdown("#### System Status")
                    st.metric("Camera", "Local video")
                    st.metric("Detection", "EAR + MAR")
                    st.metric("Alert", "Enabled")
                    risk_placeholder = st.empty()

                    if os.path.exists("incident_log.csv"):
                        st.download_button(
                            label="Download Incident Log (CSV)",
                            data=open("incident_log.csv", "rb").read(),
                            file_name="incident_log.csv",
                            mime="text/csv",
                        )
                    else:
                        st.caption("No incidents logged yet.")

                frame_counter = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    # If the video ends, restart it instead of breaking
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    frame_counter += 1
                    
                    # Only process every 2nd frame to speed up playback
                    if frame_counter % 2 != 0:
                        continue

                    processed_frame, _, current_risk = video_handler.process(frame, thresholds)
                    if current_risk is not None:
                        ear_data.append(current_risk)
                        if len(ear_data) > 50:
                            ear_data.pop(0)
                        chart_placeholder.line_chart(ear_data)
                        risk_placeholder.metric("Fatigue Risk Score", f"{round(current_risk, 1)}%")

                    placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cap.release()
        else:
            st.info("No local video file was found. Add test_video.mp4 to the project folder to enable playback.")