# ============================================================
# utils.py
# PURPOSE: Drawing helpers and visual utilities
# This is the ARTIST of our detection system
# ============================================================

import cv2        # type: ignore
import numpy as np
import time


# ------------------------------------------------------------
# FPS CALCULATOR
# Measures how many frames per second our system processes
# ------------------------------------------------------------
class FPSCounter:

    def __init__(self):
        """
        Tracks frame processing speed.
        FPS = Frames Per Second — higher = smoother video.
        """
        self.start_time  = time.time()
        self.frame_count = 0
        self.fps         = 0


    def update(self):
        """
        Call this once per frame to update FPS calculation.
        Returns current FPS as a float.
        """
        self.frame_count += 1
        elapsed           = time.time() - self.start_time

        # Recalculate FPS every second
        if elapsed >= 1.0:
            self.fps        = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time  = time.time()

        return self.fps


# ------------------------------------------------------------
# BOUNDING BOX DRAWER
# Draws professional-looking detection boxes on frames
# ------------------------------------------------------------
def draw_detection_box(frame, bbox, label, color,
                       confidence=None, track_id=None):
    """
    Draws a single beautiful bounding box with label on a frame.

    Parameters:
        frame      : OpenCV frame to draw on
        bbox       : [x1, y1, x2, y2] box coordinates
        label      : object class name (e.g. 'person')
        color      : BGR color tuple for this class
        confidence : float 0-1 (optional, shown in label)
        track_id   : integer tracking ID (optional)

    Returns:
        frame with box drawn on it
    
    WHY a separate function?
    We call this for every detected object every frame.
    Keeping it separate makes the code clean and reusable.
    """

    x1, y1, x2, y2 = bbox

    # ── Draw main bounding box rectangle ──────────────────
    cv2.rectangle(
        frame,
        (x1, y1),       # top-left corner
        (x2, y2),       # bottom-right corner
        color,
        thickness=2
    )

    # ── Build label text ───────────────────────────────────
    # Example: "person #3 87%"
    label_parts = [label]

    if track_id is not None:
        label_parts.append(f"#{track_id}")

    if confidence is not None:
        label_parts.append(f"{int(confidence * 100)}%")

    label_text = " ".join(label_parts)

    # ── Calculate label background size ───────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )

    # ── Draw filled rectangle behind text (for readability) 
    label_y = max(y1 - 10, text_h + 10)  # don't go above frame
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y),
        color,
        thickness=-1   # -1 means filled rectangle
    )

    # ── Draw label text (white on colored background) ─────
    cv2.putText(
        frame,
        label_text,
        (x1 + 2, label_y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),  # white text
        thickness
    )

    # ── Draw corner accents (professional look) ───────────
    corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_thickness = 3

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1),
             color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len),
             color, corner_thickness)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1),
             color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len),
             color, corner_thickness)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2),
             color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len),
             color, corner_thickness)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2),
             color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len),
             color, corner_thickness)

    return frame


# ------------------------------------------------------------
# STATS OVERLAY
# Draws a semi-transparent info panel on the frame
# ------------------------------------------------------------
def draw_stats_overlay(frame, fps, total_objects,
                       class_counts, mode="Detection"):
    """
    Draws a professional stats panel in the top-left corner.

    Shows:
    - Current mode (Detection or Tracking)
    - FPS (frames per second)
    - Total objects detected
    - Count per class (e.g. person: 3, car: 2)

    Parameters:
        frame        : OpenCV frame to draw on
        fps          : current frames per second
        total_objects: total detected objects this frame
        class_counts : dict of {class_name: count}
        mode         : "Detection" or "Tracking"

    Returns:
        frame with stats panel drawn
    """

    # ── Panel dimensions ───────────────────────────────────
    panel_x      = 10
    panel_y      = 10
    panel_width  = 220
    line_height  = 22
    padding      = 10

    # Calculate panel height based on number of classes shown
    max_classes  = min(len(class_counts), 6)  # show max 6 classes
    panel_height = padding * 2 + line_height * (3 + max_classes)

    # ── Draw semi-transparent background ──────────────────
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (20, 20, 20),   # very dark background
        thickness=-1
    )
    # Blend with 70% opacity
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── Draw colored top border ────────────────────────────
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + 4),
        (0, 200, 255),  # orange-yellow accent
        thickness=-1
    )

    # ── Text drawing helper ────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (220, 220, 220)  # light gray

    def put_text(text, row, color=text_color, scale=0.5):
        y_pos = panel_y + padding + (row * line_height) + 8
        cv2.putText(frame, text, (panel_x + padding, y_pos),
                    font, scale, color, 1, cv2.LINE_AA)

    # ── Draw stats content ─────────────────────────────────
    mode_color = (0, 255, 150) if mode == "Tracking" else (0, 200, 255)
    put_text(f"Mode: {mode}",          row=0, color=mode_color, scale=0.52)
    put_text(f"FPS:  {fps:.1f}",       row=1, color=(100, 255, 100))
    put_text(f"Objects: {total_objects}", row=2)

    # Divider line
    div_y = panel_y + padding + (3 * line_height)
    cv2.line(frame,
             (panel_x + 5, div_y),
             (panel_x + panel_width - 5, div_y),
             (80, 80, 80), 1)

    # Class counts
    sorted_classes = sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for idx, (class_name, count) in enumerate(sorted_classes[:6]):
        put_text(f"  {class_name}: {count}", row=4 + idx,
                 color=(180, 180, 255))

    return frame


# ------------------------------------------------------------
# FRAME PROCESSOR
# Combines detection + drawing into one clean function call
# ------------------------------------------------------------
def process_frame(frame, detections, track_history=None,
                  fps=0, mode="Detection"):
    """
    Master function that draws everything on a frame.

    Takes raw detections from detector.py and produces
    a fully annotated frame ready to display.

    Parameters:
        frame         : raw OpenCV frame
        detections    : list of dicts from detector.py
        track_history : TrackHistory object (optional)
        fps           : current FPS for display
        mode          : "Detection" or "Tracking"

    Returns:
        annotated frame ready for display
    """

    if frame is None:
        return frame

    class_counts = {}
    active_ids   = []

    for det in detections:

        bbox       = det["bbox"]
        class_name = det["class_name"]
        color      = det["color"]
        confidence = det.get("confidence", 0)
        track_id   = det.get("track_id", None)

        # ── Draw bounding box ──────────────────────────────
        draw_detection_box(
            frame, bbox, class_name, color,
            confidence=confidence,
            track_id=track_id
        )

        # ── Count classes ──────────────────────────────────
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # ── Update trail history if tracking ──────────────
        if track_history is not None and track_id is not None:
            x1, y1, x2, y2 = bbox
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_history.update(track_id, center, class_name, color)
            active_ids.append(track_id)

    # ── Draw movement trails ───────────────────────────────
    if track_history is not None:
        track_history.clean_lost_tracks(active_ids)
        frame = track_history.draw_trails(frame)

    # ── Draw stats overlay ─────────────────────────────────
    draw_stats_overlay(
        frame,
        fps          = fps,
        total_objects= len(detections),
        class_counts = class_counts,
        mode         = mode
    )

    return frame


# ------------------------------------------------------------
# FRAME RESIZE HELPER
# ------------------------------------------------------------
def resize_frame(frame, width=640):
    """
    Resizes frame to target width keeping aspect ratio.
    
    WHY: Large frames slow down detection significantly.
    640px width is the sweet spot — fast AND clear enough.
    
    Parameters:
        frame : original OpenCV frame
        width : target width in pixels (default 640)
    
    Returns:
        resized frame
    """
    if frame is None:
        return frame

    original_h, original_w = frame.shape[:2]
    aspect_ratio = original_h / original_w
    new_height   = int(width * aspect_ratio)

    return cv2.resize(frame, (width, new_height))


# ------------------------------------------------------------
# BGR TO RGB CONVERTER
# ------------------------------------------------------------
def bgr_to_rgb(frame):
    """
    Converts OpenCV BGR frame to RGB for Streamlit display.
    
    WHY: OpenCV reads images as BGR (Blue Green Red).
    Streamlit and most libraries expect RGB (Red Green Blue).
    Without this conversion, colors look wrong — red looks blue!
    
    Parameters:
        frame : BGR OpenCV frame
    
    Returns:
        RGB frame for Streamlit
    """
    if frame is None:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("TESTING utils.py")
    print("=" * 50)

    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (40, 40, 40)  # dark gray background

    # Test FPS counter
    fps_counter = FPSCounter()
    fps         = fps_counter.update()
    print(f"\n📌 FPS Counter: {fps:.1f}")

    # Test bounding box drawing
    print("\n📌 Drawing test bounding box...")
    test_frame = draw_detection_box(
        frame      = test_frame,
        bbox       = [100, 100, 300, 350],
        label      = "person",
        color      = (0, 255, 100),
        confidence = 0.92,
        track_id   = 1
    )

    # Test stats overlay
    print("📌 Drawing stats overlay...")
    test_frame = draw_stats_overlay(
        frame         = test_frame,
        fps           = 28.5,
        total_objects = 3,
        class_counts  = {"person": 2, "car": 1},
        mode          = "Tracking"
    )

    # Test resize
    resized = resize_frame(test_frame, width=320)
    print(f"\n📌 Original size : {test_frame.shape[:2]}")
    print(f"📌 Resized size  : {resized.shape[:2]}")

    # Test BGR to RGB
    rgb_frame = bgr_to_rgb(test_frame)
    print(f"\n📌 BGR→RGB conversion: ✅")

    # Save test output so you can see what it looks like
    cv2.imwrite("test_output.png", test_frame)
    print("\n📌 Saved test_output.png — open it to see the drawing!")

    print("\n" + "=" * 50)
    print("✅ utils.py tests complete!")
    print("=" * 50)