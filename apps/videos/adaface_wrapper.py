import torch
import cv2
import numpy as np
import os
from .adaface.iresnet import iresnet50
from .adaface.vit import vit_base

class AdaFaceWrapper:
    def __init__(self, model_path, device='cuda', architecture='ir50', use_alignment=True):
        # GPU fallback: if CUDA is requested but fails, use CPU
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, using CPU")
            device = 'cpu'
        self.device = device
        
        # Face Alignment (optional)
        self.use_alignment = use_alignment
        self.aligner = None
        if self.use_alignment:
            try:
                import logging
                logger = logging.getLogger(__name__)
                from .face_aligner import FaceAligner
                self.aligner = FaceAligner(device=self.device)
                logger.info("✅ FaceAligner initialized successfully - Face alignment is ENABLED")
                print("✅ FaceAligner initialized successfully - Face alignment is ENABLED")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"❌ FaceAligner initialization failed: {e}")
                logger.warning("⚠️ Falling back to simple resize - Face alignment is DISABLED")
                print(f"❌ FaceAligner initialization failed: {e}")
                print("⚠️ Falling back to simple resize - Face alignment is DISABLED")
                self.aligner = None

        # Architecture selection
        if 'vit' in os.path.basename(model_path) or architecture == 'vit':
            self.model = vit_base(num_classes=0) # num_classes=0 for feature extraction
            print("AdaFaceWrapper: Using ViT-Base architecture")
        else:
            self.model = iresnet50(num_features=512)
            print("AdaFaceWrapper: Using IR-50 architecture")

        # Load weights with error handling
        if os.path.exists(model_path):
            try:
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
            except Exception as e:
                print(f"Error loading AdaFace model: {e}")
                if self.device == 'cuda':
                    print("Retrying with CPU...")
                    self.device = 'cpu'
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"AdaFace model loaded from {model_path} on CPU")
                else:
                    raise e
        else:
            print(f"Warning: AdaFace model not found at {model_path}")

        try:
            self.model.to(self.device)
            self.model.eval()
            print(f"AdaFace model moved to {self.device}")
        except Exception as e:
            print(f"Error moving model to {self.device}, falling back to CPU: {e}")
            self.device = 'cpu'
            self.model.to(self.device)
            self.model.eval()

    def get_embedding(self, face_img, skip_alignment=False):
        """
        Extract embedding from face image.
        Args:
            face_img: BGR image (numpy array)
            skip_alignment: If True, assume face_img is already aligned 112x112
        Returns:
            embedding: 512-d numpy array (normalized)
        """
        if face_img is None or face_img.size == 0:
            return None

        try:
            # Preprocessing for AdaFace
            # 1. Face Alignment (landmark-based) or simple resize
            if not skip_alignment:
                if self.aligner is not None:
                    # Use CVLface DFA alignment
                    face_img_rgb = self.aligner.align_face(face_img)  # Returns RGB 112x112
                else:
                    # Fallback: simple resize
                    if face_img.shape[:2] != (112, 112):
                        face_img = cv2.resize(face_img, (112, 112))
                    # BGR to RGB
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                # Already aligned, just convert to RGB if needed
                if face_img.shape[:2] != (112, 112):
                    face_img = cv2.resize(face_img, (112, 112))
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 2. Continue with RGB image
            face_img = face_img_rgb
            
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
