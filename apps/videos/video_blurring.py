# -*- coding: utf-8 -*-
"""
비디오 블러 처리 모듈

이 모듈은 원본 비디오와 얼굴 정보를 받아, 지정된 얼굴을 블러 처리한 새로운 비디오를 생성합니다.
정확한 처리를 위해 YOLO Tracking과 ArcFace Embedding Matching을 사용합니다.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VideoBlurrer:
    """
    비디오 블러 처리 클래스
    
    주요 기능:
    1. 비디오 프레임 순회
    2. YOLO를 이용한 얼굴 감지 및 추적
    3. ArcFace를 이용한 얼굴 식별 (DB의 Face 모델과 매칭)
    4. 선택된 얼굴 블러 처리
    5. 결과 비디오 저장
    """
    
    def __init__(self, yolo_model_path: str, device: str = 'auto'):
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logger.info(f"VideoBlurrer initialized with device: {self.device}")

        # 1. YOLO 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        
        # 2. InsightFace 모델 로드 (매칭용)
        try:
            self.arcface_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.arcface_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(160, 160))
            logger.info("InsightFace (ArcFace) model loaded for matching")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}")
            self.arcface_app = None

    def _get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 이미지에서 임베딩 추출"""
        if self.arcface_app is None:
            return None
            
        try:
            # 이미지 크기가 너무 작으면 리사이즈
            h, w = face_img.shape[:2]
            min_size = 112
            if h < min_size or w < min_size:
                scale = max(min_size / h, min_size / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                face_img = cv2.resize(face_img, (new_w, new_h))
                
            faces = self.arcface_app.get(face_img)
            if faces:
                return faces[0].embedding
        except Exception:
            pass
        return None

    def _match_face(self, current_embedding: np.ndarray, face_models: List[Dict], threshold: float = 0.5) -> Optional[Dict]:
        """
        현재 얼굴 임베딩과 DB에 저장된 Face 모델들을 비교하여 매칭
        
        Args:
            current_embedding: 현재 프레임에서 추출한 임베딩
            face_models: DB에서 가져온 Face 정보 리스트 [{'id': ..., 'embedding': ...}, ...]
            threshold: 코사인 유사도 임계값 (0.5 이상이면 같은 사람으로 간주)
            
        Returns:
            매칭된 Face 모델 정보 또는 None
        """
        if current_embedding is None or not face_models:
            return None
            
        best_match = None
        max_similarity = -1.0
        
        for face_model in face_models:
            model_embedding = np.array(face_model['embedding'])
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                current_embedding.reshape(1, -1), 
                model_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = face_model
                
        if max_similarity >= threshold:
            return best_match
            
        return None

    def process_video(
        self, 
        video_path: str, 
        output_path: str, 
        face_models: List[Dict],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        비디오 블러 처리 실행
        
        Args:
            video_path: 원본 비디오 경로
            output_path: 저장할 비디오 경로
            face_models: Face 모델 리스트 (embedding, is_blurred 정보 포함)
            progress_callback: 진행률 업데이트 콜백 함수
            
        Returns:
            성공 여부
        """
        logger.info(f"Starting video blur processing: {video_path} -> {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
            
        # 비디오 정보 읽기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track ID별 매칭 정보 캐싱 (매 프레임 임베딩 추출 방지)
        # key: track_id, value: {'is_blurred': bool, 'last_seen': frame_idx}
        track_cache = {}
        
        frame_idx = 0
        
        # YOLO Tracking 설정
        # stream=True로 메모리 효율적 처리
        results = self.yolo_model.track(
            source=video_path,
            conf=0.5,
            persist=True,
            verbose=False,
            stream=True,
            device=self.device,
            half=(self.device == 'cuda'),  # FP16 (GPU only)
            imgsz=640
        )
        
        for result in results:
            frame = result.orig_img
            
            if result.boxes:
                for box in result.boxes:
                    if box.id is None:
                        continue
                        
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 좌표 보정
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # 블러 여부 결정
                    should_blur = True  # 기본값: 안전을 위해 블러 처리
                    
                    # 캐시 확인
                    if track_id in track_cache:
                        should_blur = track_cache[track_id]['is_blurred']
                        track_cache[track_id]['last_seen'] = frame_idx
                    else:
                        # 새로운 트랙 -> 임베딩 추출 및 매칭
                        face_img = frame[y1:y2, x1:x2]
                        embedding = self._get_embedding(face_img)
                        
                        if embedding is not None:
                            matched_face = self._match_face(embedding, face_models)
                            if matched_face:
                                should_blur = matched_face['is_blurred']
                                logger.debug(f"Frame {frame_idx}: Track {track_id} matched to Face {matched_face['id']} (Blur: {should_blur})")
                            else:
                                # 매칭 안 됨 -> 알 수 없는 얼굴 -> 블러 처리 (보수적 접근)
                                # 또는 사용자가 '선택하지 않은 얼굴'은 모두 블러 처리하라고 했으므로
                                # DB에 없는 얼굴도 블러 처리하는 것이 맞음
                                should_blur = True
                                logger.debug(f"Frame {frame_idx}: Track {track_id} not matched (Blur: {should_blur})")
                        
                        # 캐시 업데이트
                        track_cache[track_id] = {
                            'is_blurred': should_blur,
                            'last_seen': frame_idx
                        }
                    
                    # 블러 처리 적용
                    if should_blur:
                        # ROI 추출
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            # 가우시안 블러 적용
                            # 커널 크기는 ROI 크기에 비례하게 설정
                            k_w = (x2 - x1) // 3 | 1  # 홀수여야 함
                            k_h = (y2 - y1) // 3 | 1
                            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
                            frame[y1:y2, x1:x2] = blurred_roi
            
            # 프레임 저장
            out.write(frame)
            
            frame_idx += 1
            
            # 진행률 업데이트 (100프레임마다)
            if progress_callback and frame_idx % 100 == 0:
                pct = int((frame_idx / total_frames) * 100)
                progress_callback(pct)
                
            # 캐시 정리 (오래된 트랙 삭제)
            if frame_idx % 1000 == 0:
                stale_ids = [tid for tid, info in track_cache.items() if frame_idx - info['last_seen'] > 300]
                for tid in stale_ids:
                    del track_cache[tid]
                    
        cap.release()
        out.release()
        
        logger.info("Video blur processing completed")
        return True
