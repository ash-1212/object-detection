# ============================================================
# detector.py
# PURPOSE: Handles all YOLO object detection logic
# This is the EYES of our detection system
# ============================================================

# TYPE: IGNORE comments below tell Pylance to stop warning
# about these imports — they ARE installed in venv
try:
    from ultralytics import YOLO          # type: ignore
    import cvzone                          # type: ignore
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Run: pip install ultralytics cvzone")
    IMPORTS_OK = False

import cv2
import numpy as np


# ------------------------------------------------------------
# ALL 80 COCO CLASS NAMES
# Index position = class ID returned by YOLO
# ------------------------------------------------------------
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


# ------------------------------------------------------------
# COLOR MAP — unique color per class
# ------------------------------------------------------------
def generate_color_map():
    """
    Generates a unique BGR color for each of the 80 classes.
    seed=42 ensures colors are always the same every run.
    """
    np.random.seed(42)
    colors = np.random.randint(100, 255, size=(80, 3), dtype=int)
    return colors


COLOR_MAP = generate_color_map()


# ------------------------------------------------------------
# OBJECT DETECTOR CLASS
# ------------------------------------------------------------
class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Loads YOLO model once at startup.
        
        Parameters:
            model_path : YOLO weights file — auto downloads if missing
            confidence : detections below this score are ignored
        """
        self.confidence  = confidence
        self.model       = None
        self.model_ready = False

        if not IMPORTS_OK:
            print("❌ Cannot load model — missing libraries.")
            return

        self._load_model(model_path)


    def _load_model(self, model_path):
        """
        Private method to load YOLO weights.
        Called once during __init__.
        """
        try:
            self.model       = YOLO(model_path)
            self.model_ready = True
            print(f"✅ YOLO model loaded | Confidence: {self.confidence}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model_ready = False


    def detect(self, frame):
        """
        Detects objects in a single frame WITHOUT tracking.
        Use this for single images or when tracking is not needed.

        Parameters:
            frame : OpenCV image frame (numpy array BGR)

        Returns:
            list of detection dicts:
            [{bbox, confidence, class_id, class_name, color}, ...]
        """

        if frame is None or not self.model_ready:
            return []

        detections = []

        try:
            results = self.model(frame, verbose=False)

            for result in results[0].boxes:

                confidence = float(result.conf[0])
                if confidence < self.confidence:
                    continue

                class_id = int(result.cls[0])
                if class_id >= len(CLASS_NAMES):
                    continue

                coords          = result.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(c) for c in coords]
                class_name      = CLASS_NAMES[class_id]
                color           = tuple(int(c) for c in COLOR_MAP[class_id])

                detections.append({
                    "bbox"       : [x1, y1, x2, y2],
                    "confidence" : round(confidence, 2),
                    "class_id"   : class_id,
                    "class_name" : class_name,
                    "color"      : color
                })

        except Exception as e:
            print(f"Detection error: {e}")

        return detections


    def detect_with_tracking(self, frame):
        """
        Detects objects WITH ByteTrack persistent tracking IDs.
        Use this for video — gives each object a unique ID
        that stays consistent across frames.

        Parameters:
            frame : OpenCV image frame (numpy array BGR)

        Returns:
            list of tracking dicts:
            [{bbox, confidence, class_id, class_name, color, track_id}, ...]
        """

        if frame is None or not self.model_ready:
            return []

        tracked_detections = []

        try:
            results = self.model.track(
                frame,
                tracker ="bytetrack.yaml",
                persist =True,
                verbose =False
            )

            # If no tracking IDs returned this frame skip it
            if (results[0].boxes is None or
                    results[0].boxes.id is None):
                return []

            boxes       = results[0].boxes.xyxy.tolist()
            track_ids   = results[0].boxes.id.int().tolist()
            class_ids   = results[0].boxes.cls.int().tolist()
            confidences = results[0].boxes.conf.tolist()

            for box, track_id, class_id, confidence in zip(
                boxes, track_ids, class_ids, confidences
            ):
                if confidence < self.confidence:
                    continue
                if class_id >= len(CLASS_NAMES):
                    continue

                x1, y1, x2, y2 = [int(c) for c in box]
                class_name      = CLASS_NAMES[class_id]
                color           = tuple(int(c) for c in COLOR_MAP[class_id])

                tracked_detections.append({
                    "bbox"       : [x1, y1, x2, y2],
                    "confidence" : round(confidence, 2),
                    "class_id"   : class_id,
                    "class_name" : class_name,
                    "color"      : color,
                    "track_id"   : track_id
                })

        except Exception as e:
            print(f"Tracking error: {e}")

        return tracked_detections


    def get_model_info(self):
        """Returns model status info for UI display."""
        return {
            "status"    : "Loaded ✅" if self.model_ready else "Not loaded ❌",
            "classes"   : len(CLASS_NAMES),
            "threshold" : self.confidence
        }


# ------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("TESTING detector.py")
    print("=" * 50)

    detector = ObjectDetector(confidence=0.5)

    print(f"\n📌 Model Info: {detector.get_model_info()}")

    print("\n📌 Testing with blank frame...")
    blank      = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    print(f"Detections on blank frame: {len(detections)} (expected 0)")

    print("\n📌 First 5 classes:")
    for i in range(5):
        color = tuple(int(c) for c in COLOR_MAP[i])
        print(f"  ID {i}: {CLASS_NAMES[i]} → Color {color}")

    print("\n" + "=" * 50)
    print("✅ detector.py tests complete!")
    print("=" * 50)