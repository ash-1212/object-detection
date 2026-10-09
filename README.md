# 🎯 AI Object Detection & Tracking

Real-time object detection and tracking system built with 
Python, YOLOv8, OpenCV, and Streamlit.

---

## ✨ Features

- 🎯 Detects 80 object types in real time
- 🔢 Persistent tracking IDs with ByteTrack
- 🌀 Motion trail visualization
- 📊 Live FPS and class stats panel
- 📹 Supports video upload and webcam
- 🎨 Beautiful purple/lavender dark UI

---

## 🛠️ Tech Stack

| Technology   | Purpose                        |
|--------------|-------------------------------|
| Python 3.x   | Core language                  |
| YOLOv8 Nano  | Object detection model         |
| ByteTrack    | Multi-object tracking          |
| OpenCV       | Video processing               |
| Streamlit    | Web UI                         |
| cvzone       | Enhanced box drawing           |

---

## 📁 Project Structure
object-detection/
├── app.py          # Streamlit UI
├── detector.py     # YOLO detection logic
├── tracker.py      # Trail history + zone counter
├── utils.py        # Drawing + frame processing
├── requirements.txt
└── README.md

---

## ⚙️ How to Run

```bash
# 1. Clone repo
git clone https://github.com/ash-1212/object-detection.git
cd object-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
```

> The YOLOv8 model downloads automatically on first run.
---

## 📦 Auto-Downloaded Files

These files are NOT included in the repository.
They download automatically when you first run the app:

| File | Size | Source |
|------|------|--------|
| `yolov8n.pt` | ~6MB | Downloaded by Ultralytics on first run |

## 🎥 Sample Video

This project does not include sample videos due to file size.
To test the app you can:
- Use any `.mp4`, `.avi`, `.mov`, or `.mkv` video file
- Use your webcam directly from the app
- Download a free test video from https://www.pexels.com/videos/

---

## 🔄 How It Works
Video/Webcam Input
↓
OpenCV reads frames
↓
YOLOv8 detects objects
↓
ByteTrack assigns IDs
↓
utils.py draws boxes + trails
↓
Streamlit displays live output

---

## 👨‍💻 Author

- **Internship:** CodeAlpha AI Internship
