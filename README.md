# Yolo-ComputerVision-TrafficLightViolation-Detector

A comprehensive Python-based detection system using Ultralytics YOLOv8 that supports multiple input sources including cameras, video files, YouTube videos, and images with real-time parameter control.

Features
Multi-source Support: Camera, video files, YouTube URLs, and images

Real-time Control: Adjust confidence and IOU thresholds on the fly

Flexible Saving: Save images with timestamps based on various conditions

Traffic Light Detection: Special handling for red traffic light detection

Interactive Interface: Keyboard controls for parameter adjustment

Performance Monitoring: FPS counter and detection statistics


python red_TL_violationDetect.py --weights <model_path> --source <input_source> [options]

![Traffic Light Violation Demo](detection_20250828_083841_970347.png)

Input Sources
1. Camera Input
bash
# Primary webcam (default)
python red_TL_violationDetect.py --source 0

# Secondary camera
python red_TL_violationDetect.py --source 1

# With custom model and thresholds
python red_TL_violationDetect.py --source 0 --weights yolov8s.pt --conf-threshold 0.3 --iou-threshold 0.4
2. Video Files
bash
# MP4 files
python red_TL_violationDetect.py --source "video.mp4"

# AVI files
python red_TL_violationDetect.py --source "video.avi"

# MOV files
python red_TL_violationDetect.py --source "video.mov"

# MKV files
python red_TL_violationDetect.py --source "video.mkv"

# With saving options
python red_TL_violationDetect.py --source "traffic.mp4" --save-red-light --output-dir "results"
3. YouTube Videos
bash
# Full YouTube URL
python red_TL_violationDetect.py --source "https://www.youtube.com/watch?v=VIDEO_ID"

# Short YouTube URL
python red_TL_violationDetect.py --source "https://youtu.be/VIDEO_ID"

# With all options
python red_TL_violationDetect.py --source "https://youtube.com/watch?v=VIDEO_ID" --weights yolov8m.pt --conf-threshold 0.4 --save-all --show-fps
4. Image Files
bash
# JPEG images
python red_TL_violationDetect.py --source "image.jpg"

# PNG images
python red_TL_violationDetect.py --source "photo.png"

# BMP images
python red_TL_violationDetect.py --source "picture.bmp"

# WEBP images
python red_TL_violationDetect.py --source "screenshot.webp"

# With traffic light detection
python red_TL_violationDetect.py --source "traffic_light.jpg" --save-red-light --conf-threshold 0.35
Available Models
YOLOv8 comes in different sizes - choose based on your needs:

bash
# Nano (fastest, lowest accuracy)
python red_TL_violationDetect.py --weights yolov8n.pt --source 0

# Small (good balance)
python red_TL_violationDetect.py --weights yolov8s.pt --source 0

# Medium
python red_TL_violationDetect.py --weights yolov8m.pt --source 0

# Large
python red_TL_violationDetect.py --weights yolov8l.pt --source 0

# X-Large (slowest, highest accuracy)
python red_TL_violationDetect.py --weights yolov8x.pt --source 0
All Command Line Options
Option	Description	Default	Example
--weights	Path to YOLOv8 weights file	yolov8n.pt	--weights yolov8s.pt
--source	Input source (cam, file, URL, image)	0	--source "video.mp4"
--interval	Detection interval in milliseconds	100	--interval 200
--conf-threshold	Confidence threshold (0-1)	0.25	--conf-threshold 0.4
--iou-threshold	IOU threshold for NMS (0-1)	0.45	--iou-threshold 0.3
--save-all	Save all processed images	False	--save-all
--save-red-light	Save images with red traffic lights	False	--save-red-light
--output-dir	Output directory for saved images	output	--output-dir "results"
--show-fps	Display FPS counter	False	--show-fps
Real-time Keyboard Controls
While the detection is running, use these keyboard controls:

Key	Function	Description
+	Increase confidence	Increase confidence threshold by 0.05
-	Decrease confidence	Decrease confidence threshold by 0.05
]	Increase IOU	Increase IOU threshold by 0.05
[	Decrease IOU	Decrease IOU threshold by 0.05
s	Save frame	Manually save current frame
a	Toggle save all	Toggle saving all processed frames
r	Toggle red light save	Toggle saving red traffic light detections
f	Toggle FPS display	Toggle FPS counter display
q	Quit	Exit the application
Advanced Usage Scenarios
Scenario 1: Traffic Monitoring System
bash
# Monitor traffic with focus on traffic lights
python red_TL_violationDetect.py --source "highway_camera.mp4" --weights yolov8m.pt --conf-threshold 0.35 --iou-threshold 0.4 --save-red-light --output-dir "traffic_violations" --show-fps
Scenario 2: Security Camera Monitoring
bash
# Monitor security camera with high sensitivity
python red_TL_violationDetect.py --source 0 --weights yolov8s.pt --conf-threshold 0.2 --iou-threshold 0.3 --save-all --interval 50 --output-dir "security_footage"
Scenario 3: Batch Image Processing
bash
# Process multiple images (use in script)
for image in *.jpg; do
    python red_TL_violationDetect.py --source "$image" --weights yolov8l.pt --conf-threshold 0.4 --output-dir "processed_images"
done

Output Structure
Saved images are stored with timestamp filenames:

text
output/
├── detection_20231201_143045_123456.jpg
├── detection_20231201_143046_789012.jpg
└── detection_20231201_143047_345678.jpg
