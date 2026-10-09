# ============================================================
# app.py
# PURPOSE: Streamlit UI for Object Detection & Tracking
# Theme: Purple / Lavender | Fixed: sidebar text visibility
# ============================================================

import streamlit as st
import cv2             # type: ignore
import numpy as np
import tempfile
import os
import time

from detector import ObjectDetector
from tracker  import TrackHistory
from utils    import (
    process_frame,
    resize_frame,
    bgr_to_rgb,
    FPSCounter
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title = "AI Object Detection",
    page_icon  = "🎯",
    layout     = "wide"
)

# ------------------------------------------------------------
# CUSTOM CSS — Purple/Lavender Theme + Fixed Text Colors
# ------------------------------------------------------------
st.markdown("""
<style>
    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0d0b1e, #130f2e, #1a1040);
    }

    /* ── Sidebar background ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1640, #160e35) !important;
        border-right: 1px solid rgba(167,139,250,0.2) !important;
    }

    /* ── Force ALL sidebar text to white ── */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* ── Sidebar radio buttons ── */
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* ── Sidebar slider labels ── */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSlider p {
        color: #ffffff !important;
    }

    /* ── Slider track color ── */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background: #7c3aed !important;
    }

    /* ── Sidebar selectbox / input text ── */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {
        color: #000000 !important;
        background: #ffffff !important;
    }

    /* ── Sidebar divider ── */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(167,139,250,0.2) !important;
    }

    /* ── Sidebar markdown text ── */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }

    /* ── App title ── */
    .app-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #c4b5fd, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-top: 1rem;
        margin-bottom: 0.2rem;
    }

    /* ── Subtitle ── */
    .app-subtitle {
        text-align: center;
        color: rgba(196,181,253,0.55);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ── Section labels ── */
    .section-label {
        color: rgba(196,181,253,0.75);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── Info box ── */
    .info-box {
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] * {
        color: #000000 !important;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.92) !important;
        border-radius: 10px !important;
    }

    /* ── Primary button ── */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed, #6d28d9) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        opacity: 0.88 !important;
    }

    /* ── Secondary button ── */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: rgba(139,92,246,0.15) !important;
        border: 1px solid rgba(139,92,246,0.4) !important;
        border-radius: 12px !important;
        color: #c4b5fd !important;
        font-weight: 600 !important;
    }

    /* ── Success / info / warning messages ── */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(139,92,246,0.15) !important;
    }

    /* ── Caption text ── */
    .stCaption, small {
        color: rgba(196,181,253,0.55) !important;
    }
            

    /* ── Hide default Streamlit UI ── */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
def init_session_state():
    defaults = {
        "detector"      : None,
        "track_history" : None,
        "fps_counter"   : FPSCounter(),
        "total_detected": 0,
        "frame_count"   : 0,
        "class_log"     : {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------------------------
# LOAD DETECTOR
# ------------------------------------------------------------
@st.cache_resource
def load_detector(confidence):
    """Loads YOLO model once and caches it in memory."""
    return ObjectDetector(
        model_path = "yolov8n.pt",
        confidence = confidence
    )


def get_fresh_detector(confidence):
    """
    Returns detector — clears cache if mode switched to Tracking.
    WHY: ByteTrack keeps internal state. If we reuse a cached
    detector that was used for Detection, the tracking state
    is corrupted. Fresh detector = clean tracking state.
    """
    load_detector.clear()
    return load_detector(confidence)


# ------------------------------------------------------------
# STATS HTML BUILDER
# ------------------------------------------------------------
def build_stats_html(fps, current_objects, class_totals):
    """
    Compact horizontal stats bar — FPS + Objects + top classes
    all in one single row of square cards. No vertical overflow.
    """

    # Build class pill badges (top 5)
    top_classes = sorted(
        class_totals.items(),
        key     = lambda x: x[1],
        reverse = True
    )[:5]

    class_pills = ""
    for name, count in top_classes:
        class_pills += f"""
        <div style="
            background: rgba(124,58,237,0.2);
            border: 1px solid rgba(167,139,250,0.35);
            border-radius: 8px;
            padding: 6px 12px;
            text-align: center;
            min-width: 80px;">
            <div style="font-size:1rem; font-weight:700;
                        color:#c4b5fd;">{count}</div>
            <div style="font-size:0.68rem;
                        color:rgba(196,181,253,0.6);
                        white-space:nowrap">{name}</div>
        </div>"""

    no_objects = (
        '<div style="color:rgba(196,181,253,0.35);'
        'font-size:0.82rem; padding:8px 4px;">'
        'No objects detected yet</div>'
    )

    return f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        background: rgba(139,92,246,0.07);
        border: 1px solid rgba(139,92,246,0.22);
        border-radius: 14px;
        padding: 12px 16px;">

        <!-- FPS card -->
        <div style="
            background: rgba(124,58,237,0.2);
            border: 1px solid rgba(167,139,250,0.35);
            border-radius: 8px;
            padding: 6px 14px;
            text-align: center;
            min-width: 64px;">
            <div style="font-size:1rem; font-weight:800;
                        color:#a78bfa;">{fps:.0f}</div>
            <div style="font-size:0.68rem;
                        color:rgba(196,181,253,0.6);">FPS</div>
        </div>

        <!-- Objects card -->
        <div style="
            background: rgba(99,102,241,0.2);
            border: 1px solid rgba(129,140,248,0.35);
            border-radius: 8px;
            padding: 6px 14px;
            text-align: center;
            min-width: 64px;">
            <div style="font-size:1rem; font-weight:800;
                        color:#818cf8;">{current_objects}</div>
            <div style="font-size:0.68rem;
                        color:rgba(196,181,253,0.6);">Objects</div>
        </div>

        <!-- Thin divider -->
        <div style="
            width: 1px; height: 36px;
            background: rgba(167,139,250,0.2);
            flex-shrink: 0;">
        </div>

        <!-- Class pills -->
        {class_pills if class_pills else no_objects}

    </div>
    """


# ------------------------------------------------------------
# VIDEO DETECTION RUNNER
# ------------------------------------------------------------
def run_video_detection(video_path, mode, confidence,
                        frame_placeholder, stats_placeholder):
    """
    Reads video frame by frame, runs detection or tracking,
    streams annotated output to Streamlit UI.
    """

    detector      = load_detector(confidence) if mode == "Detection" else get_fresh_detector(confidence)
    track_history = TrackHistory(max_trail_length=25)
    fps_counter   = FPSCounter()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video. Please try another file.")
        return

    stop_button  = st.button(
        "⏹  Stop",
        type             = "primary",
        use_container_width = True
    )

    frame_num    = 0
    class_totals = {}

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            st.success("✅ Video processing complete!")
            break

        frame_num += 1

        
        # Skip frames only in Detection mode
        # Tracking needs every frame for accurate ID matching
        if mode == "Detection" and frame_num % 2 != 0:
            continue

        frame = resize_frame(frame, width=640)

        # Run detection or tracking based on mode
        if mode == "Tracking":
            detections = detector.detect_with_tracking(frame)
        else:
            detections = detector.detect(frame)

        fps = fps_counter.update()

        # Accumulate class counts
        for det in detections:
            name = det["class_name"]
            class_totals[name] = class_totals.get(name, 0) + 1

        # Draw all annotations on frame
        annotated = process_frame(
            frame,
            detections,
            track_history = track_history if mode == "Tracking" else None,
            fps           = fps,
            mode          = mode
        )

        # Display in Streamlit
        frame_placeholder.image(
            bgr_to_rgb(annotated),
            channels            = "RGB",
            use_container_width = True
        )

        stats_placeholder.markdown(
            build_stats_html(fps, len(detections), class_totals),
            unsafe_allow_html=True
        )

        if stop_button:
            st.info("⏹ Detection stopped.")
            break

        time.sleep(0.01)

    cap.release()


# ------------------------------------------------------------
# WEBCAM DETECTION RUNNER
# ------------------------------------------------------------
def run_webcam_detection(mode, confidence,
                         frame_placeholder, stats_placeholder):
    """Captures webcam feed and runs live detection/tracking."""

    detector      = load_detector(confidence) if mode == "Detection" else get_fresh_detector(confidence)
    track_history = TrackHistory(max_trail_length=25)
    fps_counter   = FPSCounter()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam not found. Check your camera connection.")
        return

    stop_button  = st.button(
        "⏹  Stop Webcam",
        type             = "primary",
        use_container_width = True
    )

    class_totals = {}

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            st.error("❌ Lost webcam feed.")
            break

        frame = resize_frame(frame, width=640)

        if mode == "Tracking":
            detections = detector.detect_with_tracking(frame)
        else:
            detections = detector.detect(frame)

        fps = fps_counter.update()

        for det in detections:
            name = det["class_name"]
            class_totals[name] = class_totals.get(name, 0) + 1

        annotated = process_frame(
            frame, detections,
            track_history = track_history if mode == "Tracking" else None,
            fps           = fps,
            mode          = mode
        )

        frame_placeholder.image(
            bgr_to_rgb(annotated),
            channels            = "RGB",
            use_container_width = True
        )

        stats_placeholder.markdown(
            build_stats_html(fps, len(detections), class_totals),
            unsafe_allow_html=True
        )

        if stop_button:
            st.info("⏹ Webcam stopped.")
            break

        time.sleep(0.01)

    cap.release()


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
def main():

    init_session_state()

    # ── Title ──────────────────────────────────────────────
    st.markdown(
        '<div class="app-title">🎯 AI Object Detection & Tracking</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="app-subtitle">'
        'Real-time detection powered by YOLOv8 · ByteTrack · OpenCV'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:

        st.markdown(
            '<p style="font-size:1.2rem; font-weight:700;'
            'color:#c4b5fd; margin-bottom:0.5rem;">⚙️ Controls</p>',
            unsafe_allow_html=True
        )
        st.divider()

        # Input source
        st.markdown(
            '<p style="font-size:0.78rem; font-weight:600;'
            'letter-spacing:0.1em; text-transform:uppercase;'
            'color:#c4b5fd;">📥 Input Source</p>',
            unsafe_allow_html=True
        )
        source = st.radio(
            "Source",
            options          = ["Upload Video", "Webcam"],
            label_visibility = "collapsed"
        )

        st.divider()

        # Detection mode
        st.markdown(
            '<p style="font-size:0.78rem; font-weight:600;'
            'letter-spacing:0.1em; text-transform:uppercase;'
            'color:#c4b5fd;">🔧 Mode</p>',
            unsafe_allow_html=True
        )
        mode = st.radio(
            "Mode",
            options          = ["Detection", "Tracking"],
            label_visibility = "collapsed"
        )

        st.markdown(
            '<div class="info-box">'
            '🔵 <b>Detection</b> — finds objects per frame<br><br>'
            '🟢 <b>Tracking</b> — persistent IDs + motion trails'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Confidence slider
        st.markdown(
            '<p style="font-size:0.78rem; font-weight:600;'
            'letter-spacing:0.1em; text-transform:uppercase;'
            'color:#c4b5fd;">🎯 Confidence Threshold</p>',
            unsafe_allow_html=True
        )
        confidence = st.slider(
            "Confidence",
            min_value        = 0.10,
            max_value        = 0.95,
            value            = 0.50,
            step             = 0.05,
            label_visibility = "collapsed"
        )
        st.caption(f"Minimum score: {int(confidence * 100)}%")

        st.divider()

        # Model info
        st.markdown(
            '<p style="font-size:0.78rem; font-weight:600;'
            'letter-spacing:0.1em; text-transform:uppercase;'
            'color:#c4b5fd;">📊 Model Info</p>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
- **Model:** YOLOv8 Nano
- **Classes:** 80 objects
- **Tracker:** ByteTrack
- **Threshold:** {int(confidence * 100)}%
        """)

    # ── Main layout: video + stats ─────────────────────────
    # ── Full width video output ────────────────────────────
    st.markdown(
        '<div class="section-label">📹 Video Output</div>',
        unsafe_allow_html=True
    )
    frame_placeholder = st.empty()
    frame_placeholder.markdown(
        '<div style="background:rgba(139,92,246,0.05);'
        'border:2px dashed rgba(139,92,246,0.25);'
        'border-radius:16px; height:360px;'
        'display:flex; align-items:center;'
        'justify-content:center;'
        'color:rgba(196,181,253,0.3); font-size:1rem;">'
        '🎯 Video feed will appear here'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Compact stats row (3 equal boxes) ─────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:1rem">'
        '📈 Live Stats</div>',
        unsafe_allow_html=True
    )
    stats_placeholder = st.empty()
    stats_placeholder.markdown(
        build_stats_html(0, 0, {}),
        unsafe_allow_html=True
    )

    st.divider()

    # ── Upload Video ───────────────────────────────────────
    if source == "Upload Video":

        st.markdown(
            '<div class="section-label">📂 Upload Video File</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload video",
            type             = ["mp4", "avi", "mov", "mkv"],
            label_visibility = "collapsed"
        )

        if uploaded is not None:

            # Save to temp file — OpenCV needs a file path
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.success(f"✅ Ready: **{uploaded.name}**"
                       f"  ({round(uploaded.size/1024/1024, 1)} MB)")

            button_label = "▶️  Start Detection" if mode == "Detection" else "🔢  Start Tracking"

            start = st.button(
                button_label,
                type             = "primary",
                use_container_width = True
            )

            if start:
                with st.spinner("Loading YOLO model..."):
                    pass
                run_video_detection(
                    video_path        = tmp_path,
                    mode              = mode,
                    confidence        = confidence,
                    frame_placeholder = frame_placeholder,
                    stats_placeholder = stats_placeholder
                )
                os.unlink(tmp_path)

    # ── Webcam ─────────────────────────────────────────────
    else:

        st.markdown(
            '<div class="info-box">'
            '📷 Make sure your webcam is connected and '
            'not in use by another app before starting.'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "📷  Start Webcam",
            type             = "primary",
            use_container_width = True
        ):
            run_webcam_detection(
                mode              = mode,
                confidence        = confidence,
                frame_placeholder = frame_placeholder,
                stats_placeholder = stats_placeholder
            )

    # ── Footer ─────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="text-align:center;'
        'color:rgba(196,181,253,0.25);'
        'font-size:0.78rem; padding-bottom:1rem;">'
        'Built with ❤️ using Python · YOLOv8 · OpenCV · Streamlit'
        '<br>CodeAlpha AI Internship Project</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()