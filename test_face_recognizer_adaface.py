import os
import sys
import django
from django.conf import settings

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

import numpy as np
from apps.videos.face_recognizer import FaceRecognizer

def test_face_recognizer_adaface():
    print("Testing FaceRecognizer with AdaFace...")
    try:
        # Initialize FaceRecognizer with AdaFace
        recognizer = FaceRecognizer(model_name='adaface', device='cpu')
        print("FaceRecognizer initialized successfully.")
        
        # Create a dummy image (simulating a face crop)
        # Size doesn't have to be 112x112, as AdaFaceWrapper resizes it.
        # Let's use a larger size to simulate a real crop.
        dummy_face = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        
        # Get embedding
        emb = recognizer.get_embedding(dummy_face)
        
        if emb is not None:
            print(f"Embedding extracted successfully. Shape: {emb.shape}")
            print(f"First 5 values: {emb[:5]}")
        else:
            print("Failed to extract embedding (returned None).")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_recognizer_adaface()
