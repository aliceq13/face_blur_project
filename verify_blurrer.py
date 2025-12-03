import os
import django
import logging
import sys
import numpy as np
import cv2

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

from apps.videos.video_blurring import VideoBlurrer
from django.conf import settings

# Configure logging to stdout
# Force root logger to INFO and add stdout handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(handler)

# Ensure apps.videos logger is also set
logging.getLogger('apps.videos').setLevel(logging.INFO)
logging.getLogger('apps.videos').addHandler(handler)

logger = logging.getLogger(__name__)

def verify():
    # Use the video found in the workspace
    video_path = 'media/videos/8ef99a9b-49df-45ec-9e70-3a5c5b184355.mp4'
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        # Try absolute path if relative fails (though in container /app is root)
        video_path = '/app/media/videos/8ef99a9b-49df-45ec-9e70-3a5c5b184355.mp4'
        if not os.path.exists(video_path):
             logger.error(f"Video still not found: {video_path}")
             return

    logger.info(f"Starting verification on {video_path}")
    print(f"DEBUG: Checking video path: {os.path.abspath(video_path)}")
    if os.path.exists(video_path):
        print("DEBUG: File exists.")
    else:
        print("DEBUG: File DOES NOT exist.")
        
    # Test cv2 open
    cap_test = cv2.VideoCapture(video_path)
    if not cap_test.isOpened():
        print("DEBUG: cv2 failed to open video.")
    else:
        print(f"DEBUG: cv2 opened video. Frames: {cap_test.get(cv2.CAP_PROP_FRAME_COUNT)}")
    cap_test.release()
    
    # Initialize Blurrer
    # Assuming model path is at /app/models/yolov11s-face.pt
    model_path = 'models/yolov11s-face.pt'
    if not os.path.exists(model_path):
        model_path = '/app/models/yolov11s-face.pt'
        
    blurrer = VideoBlurrer(
        yolo_model_path=model_path,
        device='auto'
    )
    
    # Extract a target face from the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Failed to read video")
        return

    logger.info("Detecting face in the first 100 frames to use as target...")
    # Use YOLO directly to find a face
    target_embedding = None
    
    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = blurrer.yolo_model(frame, verbose=False)
        
        for result in results:
            if result.boxes:
                # Pick the largest face
                best_box = None
                max_area = 0
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = box
                
                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    # Ensure face is large enough
                    if (x2-x1) > 30 and (y2-y1) > 30:
                        face_img = frame[y1:y2, x1:x2]
                        target_embedding = blurrer._get_embedding(face_img)
                        if target_embedding is not None:
                            logger.info(f"SUCCESS: Found target face at frame {frame_count}, coords {x1},{y1},{x2},{y2}")
                            break
        
        if target_embedding is not None:
            break
        frame_count += 1
    
    cap.release()
    
    if target_embedding is None:
        logger.error("No face found in first 100 frames to use as target.")
        return
        
    logger.info("Found target face.")
    
    # Construct face_model
    face_models = [{
        'id': 1,
        'embedding': target_embedding.tolist(),
        'embeddings': [], # Start empty
        'is_blurred': True
    }]
    
    logger.info("Step 2: Run analysis with target face...")
    raw_tracks, track_decisions, meta = blurrer._analyze_video(video_path, face_models=face_models)
    
    # Check results
    identified_count = sum(1 for d in track_decisions.values() if d['face_id'] == 1)
    logger.info(f"Identified {identified_count} tracks as Target Face.")
    
    # Check for Query Expansion logs in stdout
    logger.info("Verification Complete.")

if __name__ == '__main__':
    verify()
