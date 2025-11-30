import torch
import cv2
import numpy as np
import os
from .adaface.iresnet import iresnet50
from .adaface.vit import vit_base

class AdaFaceWrapper:
    def __init__(self, model_path, device='cuda', architecture='ir50'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Architecture selection
        if 'vit' in os.path.basename(model_path) or architecture == 'vit':
            self.model = vit_base(num_classes=0) # num_classes=0 for feature extraction
            print("AdaFaceWrapper: Using ViT-Base architecture")
        else:
            self.model = iresnet50(num_features=512)
            print("AdaFaceWrapper: Using IR-50 architecture")
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle state_dict key if present
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"AdaFace model loaded from {model_path}")
        else:
            print(f"Warning: AdaFace model not found at {model_path}")
            
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, face_img):
        """
        Extract embedding from face image.
        Args:
            face_img: BGR image (numpy array)
        Returns:
            embedding: 512-d numpy array (normalized)
        """
        if face_img is None or face_img.size == 0:
            return None
            
        try:
            # Preprocessing for AdaFace
            # 1. Resize to 112x112
            face_img = cv2.resize(face_img, (112, 112))
            
            # 2. BGR to RGB? 
            # AdaFace official repo uses BGR if using cv2.imread, but usually models trained with PIL use RGB.
            # However, the official repo's inference_demo.py uses cv2.imread (BGR) and converts to RGB?
            # Let's check standard practice. Most InsightFace/ArcFace models use RGB.
            # AdaFace repo: "img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"
            # So we should convert to RGB.
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize ((img / 255.0) - 0.5) / 0.5  == (img - 127.5) / 127.5
            face_img = ((face_img / 255.0) - 0.5) / 0.5
            
            # 4. Transpose (H, W, C) -> (C, H, W)
            face_img = face_img.transpose((2, 0, 1))
            
            # 5. To Tensor
            face_tensor = torch.from_numpy(face_img).float().unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                embedding = self.model(face_tensor)
            
            # ✅ CRITICAL FIX: L2 정규화 (Cosine similarity를 위해 필수)
            embedding = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"AdaFace inference failed: {e}")
            return None
