"""
Face Alignment Module using CVLface DFA (Deep Face Alignment)

This module provides landmark-based face alignment to normalize face orientation
before embedding extraction, significantly improving recognition accuracy.
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import logging

logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Face alignment using CVLface DFA MobileNet model.
    Aligns faces to canonical 112x112 position using facial landmarks.
    """
    
    def __init__(self, device='auto', model_id='minchul/cvlface_DFA_mobilenet'):
        """
        Initialize Face Aligner.
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
            model_id: HuggingFace model ID for alignment model
        """
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model_id = model_id
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load alignment model from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
            import sys
            
            # Download model files
            cache_dir = os.path.expanduser(f'~/.cvlface_cache/{self.model_id}')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Download model weights
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename='model.pth',
                cache_dir=cache_dir,
                token=os.environ.get('HF_TOKEN', None)
            )
            
            # Load CVLface model architecture
            # Note: This requires CVLface to be in the Python path
            cvlface_path = os.path.join(os.path.dirname(__file__), '../../CVLface')
            if os.path.exists(cvlface_path):
                sys.path.insert(0, cvlface_path)
            
            try:
                from cvlface.research.recognition.code.run_v1.models.dfa_mobilenet import DFAMobileNet
                self.model = DFAMobileNet()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"FaceAligner loaded successfully on {self.device}")
            except ImportError as e:
                logger.warning(f"Could not import CVLface DFA model: {e}")
                logger.warning("Falling back to simple resize alignment")
                self.model = None
                
        except Exception as e:
            logger.warning(f"Failed to load alignment model: {e}")
            logger.warning("Falling back to simple resize alignment")
            self.model = None
    
    def _pil_to_input(self, pil_image):
        """Convert PIL image to model input tensor"""
        trans = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans(pil_image).unsqueeze(0).to(self.device)
    
    def _visualize(self, tensor):
        """Convert tensor back to PIL image"""
        # Denormalize
        tensor = tensor.clone()
        tensor = tensor * 0.5 + 0.5  # Reverse normalization
        tensor = tensor.clamp(0, 1)
        
        # To PIL
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def align_face(self, face_img_bgr):
        """
        Align face to canonical position.
        
        Args:
            face_img_bgr: Face image in BGR format (numpy array)
            
        Returns:
            aligned_face_rgb: Aligned face in RGB format (112x112 numpy array)
        """
        if self.model is None:
            # Fallback: simple resize
            return self._fallback_align(face_img_bgr)
        
        try:
            # Convert BGR to RGB
            face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize to expected input size (e.g., 112x112 or larger)
            # DFA models typically expect 112x112 input
            if face_img_rgb.shape[:2] != (112, 112):
                face_img_rgb = cv2.resize(face_img_rgb, (112, 112))
            
            # Convert to PIL
            pil_img = Image.fromarray(face_img_rgb)
            
            # Prepare input
            input_tensor = self._pil_to_input(pil_img)
            
            # Align
            with torch.no_grad():
                aligned_tensor, orig_pred_ldmks, aligned_ldmks, score, thetas, normalized_bbox = self.model(input_tensor)
            
            # Convert back to numpy (RGB)
            aligned_pil = self._visualize(aligned_tensor)
            aligned_rgb = np.array(aligned_pil)
            
            return aligned_rgb
            
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, falling back to resize")
            return self._fallback_align(face_img_bgr)
    
    def _fallback_align(self, face_img_bgr):
        """Fallback: simple resize to 112x112"""
        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        if face_img_rgb.shape[:2] != (112, 112):
            face_img_rgb = cv2.resize(face_img_rgb, (112, 112))
        return face_img_rgb
