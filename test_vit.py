import torch
import os
import cv2
import numpy as np
from apps.videos.adaface_wrapper import AdaFaceWrapper

def test_vit():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Check for .pt file first
    weight_path = os.path.join(base_dir, 'apps', 'videos', 'weights', 'adaface_vit_base_kprpe_webface4m.pt')
    
    print(f"Testing ViT model loading from: {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"File not found: {weight_path}")
        return

    try:
        wrapper = AdaFaceWrapper(weight_path, device='cpu', architecture='vit')
        print("Model loaded successfully!")
        
        # Test inference
        dummy_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emb = wrapper.get_embedding(dummy_img)
        
        if emb is not None:
            print(f"Inference successful! Embedding shape: {emb.shape}")
            print(f"First 5 values: {emb[:5]}")
        else:
            print("Inference returned None")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vit()
