import argparse
import time
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import pafy
import urllib.parse
import os

class YOLOv8Detector:
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLOv8 detector
        
        Args:
            weights_path: Path to YOLOv8 weights file
            conf_threshold: Confidence threshold for detection (0-1)
            iou_threshold: IOU overlap threshold for NMS (0-1)
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        
    def set_confidence_threshold(self, conf_threshold):
        """Set confidence threshold"""
        self.conf_threshold = conf_threshold
        
    def set_iou_threshold(self, iou_threshold):
        """Set IOU threshold"""
        self.iou_threshold = iou_threshold
        
    def detect(self, frame):
        """Perform detection on a single frame"""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            max_det=100
        )
        return results[0]  # Return first result (single image)

def is_image_file(file_path):
    """Check if the file is an image based on extension"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def load_image(image_path):
    """Load an image file"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    return image

def setup_source(source):
    """
    Setup source from camera, file, YouTube URL, or image
    
    Returns:
        tuple: (source_type, cap, image) where:
            source_type: 'camera', 'video', 'youtube', 'image', or None
            cap: cv2.VideoCapture object (for video/camera) or None
            image: loaded image (for image) or None
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        # Camera source
        cap = cv2.VideoCapture(int(source))
        if cap.isOpened():
            return 'camera', cap, None
        else:
            print(f"Error: Could not open camera: {source}")
            return None, None, None
            
    elif isinstance(source, str) and source.startswith(('http://', 'https://')):
        # YouTube URL or web URL
        if 'youtube.com' in source or 'youtu.be' in source:
            try:
                # Try to parse as YouTube URL
                video = pafy.new(source)
                best = video.getbest(preftype="mp4")
                cap = cv2.VideoCapture(best.url)
                if cap.isOpened():
                    print(f"Playing YouTube video: {video.title}")
                    return 'youtube', cap, None
                else:
                    print("Error: Could not open YouTube stream")
            except Exception as e:
                print(f"Error loading YouTube video: {e}")
                print("Trying as direct video URL...")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    return 'video', cap, None
        else:
            # Other web URL
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return 'video', cap, None
    
    elif isinstance(source, str):
        # Check if it's an image file
        if is_image_file(source):
            image = load_image(source)
            if image is not None:
                print(f"Loaded image: {source}")
                return 'image', None, image
            else:
                return None, None, None
        else:
            # Video file
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return 'video', cap, None
    
    # If we get here, the source couldn't be opened
    print(f"Error: Could not open source: {source}")
    return None, None, None

def save_image_with_timestamp(frame, output_dir="output"):
    """Save image with timestamp filename"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{output_dir}/detection_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")
    return filename

def is_traffic_light_red(detections, class_names, conf_threshold=0.5):
    """
    Check if a red traffic light is detected
    Adjust class names based on your model's training
    """
    for detection in detections.boxes:
        if detection.conf.item() >= conf_threshold:
            class_id = int(detection.cls.item())
            class_name = class_names[class_id]
            
            # Adjust these class names based on your model's training
            traffic_light_classes = ['traffic light', 'traffic_light', 'red light', 'red_light', 'light']
            if any(tl_class in class_name.lower() for tl_class in traffic_light_classes):
                return True
    return False

def draw_detections(frame, detections, class_names, fps=None, params=None):
    """Draw detection boxes and labels on frame"""
    result_frame = frame.copy()
    
    if detections.boxes is not None:
        for detection in detections.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            conf = detection.conf.item()
            class_id = int(detection.cls.item())
            class_name = class_names[class_id]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display current parameters if provided
    if params:
        param_text = f"Conf: {params['conf']:.2f} | IOU: {params['iou']:.2f}"
        cv2.putText(result_frame, param_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if fps is not None:
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display controls help
    controls = "Controls: +/- conf | [/] iou | s: save | q: quit"
    cv2.putText(result_frame, controls, (10, result_frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_frame

def print_controls():
    """Print available keyboard controls"""
    print("\n=== Keyboard Controls ===")
    print("+ : Increase confidence threshold")
    print("- : Decrease confidence threshold")
    print("[ : Decrease IOU threshold")
    print("] : Increase IOU threshold")
    print("s : Save current frame")
    print("a : Toggle save all frames")
    print("r : Toggle save red light frames")
    print("f : Toggle FPS display")
    print("q : Quit")
    print("=========================\n")

def process_image_mode(detector, image, args):
    """Process a single image in interactive mode"""
    global save_all_frames, save_red_light_frames
    
    print("Image mode: Press keys to adjust parameters, 's' to save, 'q' to quit")
    
    current_image = image.copy()
    results = detector.detect(current_image)
    rendered_image = draw_detections(current_image, results, detector.class_names, params={
        'conf': detector.conf_threshold,
        'iou': detector.iou_threshold
    })
    
    red_light_detected = is_traffic_light_red(results, detector.class_names)
    if red_light_detected:
        print("Red traffic light detected in image!")
    
    while True:
        cv2.imshow('YOLOv8 Image Detection - Controls: +- conf, [] iou, s: save, q: quit', rendered_image)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            # Increase confidence threshold
            new_conf = min(0.95, detector.conf_threshold + 0.05)
            detector.set_confidence_threshold(new_conf)
            print(f"Confidence threshold: {new_conf:.2f}")
        elif key == ord('-'):
            # Decrease confidence threshold
            new_conf = max(0.05, detector.conf_threshold - 0.05)
            detector.set_confidence_threshold(new_conf)
            print(f"Confidence threshold: {new_conf:.2f}")
        elif key == ord(']'):
            # Increase IOU threshold
            new_iou = min(0.95, detector.iou_threshold + 0.05)
            detector.set_iou_threshold(new_iou)
            print(f"IOU threshold: {new_iou:.2f}")
        elif key == ord('['):
            # Decrease IOU threshold
            new_iou = max(0.05, detector.iou_threshold - 0.05)
            detector.set_iou_threshold(new_iou)
            print(f"IOU threshold: {new_iou:.2f}")
        elif key == ord('s'):
            # Save current image
            save_image_with_timestamp(rendered_image, args.output_dir)
            print("Image saved")
        elif key == ord('a'):
            # Toggle save all (not applicable for single image)
            save_all_frames = not save_all_frames
            status = "ENABLED" if save_all_frames else "DISABLED"
            print(f"Save all frames: {status} (not applicable for single image)")
        elif key == ord('r'):
            # Toggle save red light
            save_red_light_frames = not save_red_light_frames
            status = "ENABLED" if save_red_light_frames else "DISABLED"
            print(f"Save red light frames: {status}")
        
        # Re-detect with new parameters
        results = detector.detect(image)
        rendered_image = draw_detections(image, results, detector.class_names, params={
            'conf': detector.conf_threshold,
            'iou': detector.iou_threshold
        })
        
        # Check for red traffic light
        red_light_detected = is_traffic_light_red(results, detector.class_names)
        if red_light_detected:
            print("Red traffic light detected!")
            if save_red_light_frames:
                save_image_with_timestamp(rendered_image, args.output_dir)
                print("Red traffic light image saved!")
    
    cv2.destroyAllWindows()

# Global variables for parameter control
detector = None
save_all_frames = False
save_red_light_frames = False
show_fps = False

def main():
    global detector, save_all_frames, save_red_light_frames, show_fps
    
    parser = argparse.ArgumentParser(description='YOLOv8 Detection with support for Camera, Video, YouTube, and Images')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to weights file')
    parser.add_argument('--source', type=str, default='0', help='Camera index (0), video file, YouTube URL, or image file')
    parser.add_argument('--interval', type=float, default=100, help='Detection interval in milliseconds (for video/camera)')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Initial confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='Initial IOU threshold')
    parser.add_argument('--save-all', action='store_true', help='Initially save all images')
    parser.add_argument('--save-red-light', action='store_true', help='Initially save red light images')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for saved images')
    parser.add_argument('--show-fps', action='store_true', help='Initially show FPS counter')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv8Detector(
        weights_path=args.weights,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Set initial flags
    save_all_frames = args.save_all
    save_red_light_frames = args.save_red_light
    show_fps = args.show_fps
    
    # Setup source
    source_type, cap, image = setup_source(args.source)
    if source_type is None:
        print(f"Error: Could not open source: {args.source}")
        return
    
    print(f"Source type: {source_type}")
    print(f"Initial confidence threshold: {args.conf_threshold}")
    print(f"Initial IOU threshold: {args.iou_threshold}")
    print(f"Save all frames: {save_all_frames}")
    print(f"Save red light frames: {save_red_light_frames}")
    print(f"Show FPS: {show_fps}")
    print_controls()
    
    # Handle image mode separately
    if source_type == 'image':
        process_image_mode(detector, image, args)
        return
    
    # Video/camera mode
    last_detection_time = 0
    detection_count = 0
    fps_counter = 0
    fps_time = time.time()
    fps_value = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            current_time = time.time() * 1000  # Convert to milliseconds
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_value = fps_counter
                fps_counter = 0
                fps_time = time.time()
                if show_fps:
                    print(f"FPS: {fps_value}")
            
            # Check if it's time for detection
            if current_time - last_detection_time >= args.interval:
                # Perform detection
                results = detector.detect(frame)
                
                # Draw detections on frame
                rendered_frame = draw_detections(frame, results, detector.class_names, 
                                               fps_value if show_fps else None,
                                               {'conf': detector.conf_threshold, 'iou': detector.iou_threshold})
                
                # Check for red traffic light
                red_light_detected = False
                if save_red_light_frames:
                    red_light_detected = is_traffic_light_red(results, detector.class_names)
                
                # Save images based on conditions
                if save_all_frames:
                    save_image_with_timestamp(rendered_frame, args.output_dir)
                elif save_red_light_frames and red_light_detected:
                    save_image_with_timestamp(rendered_frame, args.output_dir)
                    print("Red traffic light detected! Image saved.")
                
                last_detection_time = current_time
                detection_count += 1
                
                # Display frame with detections
                cv2.imshow('YOLOv8 Detection - Controls: +- conf, [] iou, s: save, q: quit', rendered_frame)
            else:
                # Display frame without new detection
                cv2.imshow('YOLOv8 Detection - Controls: +- conf, [] iou, s: save, q: quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                # Increase confidence threshold
                new_conf = min(0.95, detector.conf_threshold + 0.05)
                detector.set_confidence_threshold(new_conf)
                print(f"Confidence threshold: {new_conf:.2f}")
            elif key == ord('-'):
                # Decrease confidence threshold
                new_conf = max(0.05, detector.conf_threshold - 0.05)
                detector.set_confidence_threshold(new_conf)
                print(f"Confidence threshold: {new_conf:.2f}")
            elif key == ord(']'):
                # Increase IOU threshold
                new_iou = min(0.95, detector.iou_threshold + 0.05)
                detector.set_iou_threshold(new_iou)
                print(f"IOU threshold: {new_iou:.2f}")
            elif key == ord('['):
                # Decrease IOU threshold
                new_iou = max(0.05, detector.iou_threshold - 0.05)
                detector.set_iou_threshold(new_iou)
                print(f"IOU threshold: {new_iou:.2f}")
            elif key == ord('s'):
                # Save current frame
                save_image_with_timestamp(frame, args.output_dir)
                print("Manual save triggered")
            elif key == ord('a'):
                # Toggle save all frames
                save_all_frames = not save_all_frames
                status = "ENABLED" if save_all_frames else "DISABLED"
                print(f"Save all frames: {status}")
            elif key == ord('r'):
                # Toggle save red light frames
                save_red_light_frames = not save_red_light_frames
                status = "ENABLED" if save_red_light_frames else "DISABLED"
                print(f"Save red light frames: {status}")
            elif key == ord('f'):
                # Toggle FPS display
                show_fps = not show_fps
                status = "ENABLED" if show_fps else "DISABLED"
                print(f"FPS display: {status}")
                
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print(f"Total detections performed: {detection_count}")

if __name__ == "__main__":
    main()