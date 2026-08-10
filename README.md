# AI-Project

IVP and AI Project

## HelpGuard AI

## Project Overview
HelpGuard AI is a lightweight, explainable computer vision project for monitoring fatigue-related facial cues from a local video source. The application combines MediaPipe Face Mesh, OpenCV, NumPy, and Streamlit to estimate Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) frame by frame. It then presents the results through a polished dashboard that explains the reasoning behind each alert in a simple, academic-friendly way.

## Why this project matters
This project is suitable for a B.Tech Computer Science minor project because it demonstrates how a real AI-like system can be built locally using classical computer vision techniques. It is simple to explain in a viva, easy to run, and still shows meaningful engineering choices such as modular design, threshold-based logic, explainability, and local logging.

## Main Features
- Local video analysis using OpenCV
- Facial landmark tracking using MediaPipe Face Mesh
- EAR and MAR estimation for fatigue monitoring
- Explainable alert reasons for fatigue-related facial cues
- Live monitoring summary statistics
- CSV-based incident logging
- Clean Streamlit dashboard with polished UI

## Installation
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run main.py
```

## Demo Instructions
1. Open the Streamlit dashboard.
2. Upload a local video or use the sample file.
3. Adjust the EAR and MAR thresholds.
4. Click Run analysis.
5. Review the annotated frame, summary metrics, explanation panel, and incident log.

## Folder Structure
- main.py: application entry point
- drowsy_detection.py: compatibility wrapper for the detector module
- app/: modular package containing the dashboard, detection logic, metrics helpers, and incident logging
- test_video.mp4: sample video file for quick testing
- requirements.txt: project dependencies

## Architecture Diagram
```text
User Input (video file)
        |
        v
Streamlit Dashboard
        |
        v
OpenCV Frame Reader
        |
        v
MediaPipe Face Mesh
        |
        v
EAR and MAR Analysis
        |
        v
Alert Explanation + Metrics + CSV Logging
```

## How AI is used
The project uses MediaPipe Face Mesh to detect facial landmarks from each frame. From these landmarks, it estimates:
- EAR (Eye Aspect Ratio) to represent eye openness
- MAR (Mouth Aspect Ratio) to represent mouth openness

These measurements are then compared against thresholds to provide an explainable fatigue-like alert.

## Screenshots Placeholder
- Dashboard overview: coming soon
- Annotated frame view: coming soon
- Explainability section: coming soon

## Future Scope
Possible extensions include:
- Multi-person tracking
- Posture and fall monitoring
- Gesture-based distress detection
- Better alert visualization
- Report generation in PDF

## Limitations
- The system is based on simple threshold logic rather than a deep learning model.
- Performance depends on the quality of the input video and facial visibility.
- It is meant for academic demonstration and prototype monitoring rather than production deployment.

## Technologies Used
- Python
- Streamlit
- OpenCV
- MediaPipe
 - NumPy
