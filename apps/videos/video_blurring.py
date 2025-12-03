# -*- coding: utf-8 -*-
"""
비디오 블러 처리 모듈 (YOLO Face v11s + AdaFace ViT-12M + Efficient Single-Pass)

이 모듈은 원본 비디오와 얼굴 정보를 받아, 지정된 얼굴을 블러 처리한 새로운 비디오를 생성합니다.

핵심 기술:
1. YOLO Face v11s: 빠른 실시간 얼굴 감지
2. AdaFace ViT-12M: 고정밀 얼굴 인식
3. Multi-Embedding Matching: 여러 임베딩으로 정확도 향상
4. Efficient Single-Pass: 한 번의 순회로 처리 완료
5. 적응형 블러: 얼굴 크기에 따른 동적 블러 강도
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional
import logging
import os
import subprocess
import shutil
import gc

logger = logging.getLogger(__name__)


class VideoBlurrer:
    """
    비디오 블러 처리 클래스

    매 프레임마다:
    1. YOLO로 얼굴 감지
    2. AdaFace로 임베딩 추출
    3. 타겟과 비교 (Multi-Embedding Matching)
    4. 타겟이 아니면 블러 적용
    5. 프레임 저장
    """

    def __init__(
        self,
        yolo_model_path: str,
        device: str = 'auto',
        threshold: float = 0.92,  # 매우 엄격한 매칭 (0.92)
        use_multi_embedding: bool = False  # Multi-Embedding 비활성화 (메인만 사용)
    ):
        self.threshold = threshold
        self.use_multi_embedding = use_multi_embedding

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"🚀 Initializing VideoBlurrer on {self.device}")

        # 1. YOLO Face v11s 로드
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"✅ YOLO Face model loaded: {yolo_model_path}")

        # 2. AdaFace ViT-12M 로드
        try:
            from .adaface_wrapper import AdaFaceWrapper
            from django.conf import settings

            model_path = str(getattr(settings, 'ADAFACE_MODEL_PATH', None))
            model_arch = getattr(settings, 'ADAFACE_ARCHITECTURE', 'ir_101')

            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"AdaFace model not found: {model_path}")

            self.face_recognizer = AdaFaceWrapper(
                model_path=model_path,
                architecture=model_arch,
                device=self.device
            )
            logger.info(f"✅ AdaFace {model_arch} loaded from {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load AdaFace: {e}")
            self.face_recognizer = None

    def _get_embedding(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 이미지에서 임베딩 추출"""
        if self.face_recognizer is None:
            return None
        return self.face_recognizer.get_embedding(face_img_bgr)

    def _is_target_face(
        self,
        face_img_bgr: np.ndarray,
        target_embeddings: List[np.ndarray],
        multi_embeddings: List[List[np.ndarray]],
        frame_idx: int = 0
    ) -> tuple:
        """
        얼굴이 타겟인지 확인 (Multi-Embedding Matching)

        Args:
            face_img_bgr: 얼굴 이미지 (BGR)
            target_embeddings: 타겟의 메인 임베딩 리스트
            multi_embeddings: 타겟의 Multi-Thumbnail 임베딩 리스트
            frame_idx: 현재 프레임 번호 (로깅용)

        Returns:
            (is_target, max_sim): (타겟 여부, 최대 유사도)
        """
        if (not target_embeddings and not multi_embeddings) or face_img_bgr is None or face_img_bgr.size == 0:
            return False, 0.0

        # 임베딩 추출
        embedding = self._get_embedding(face_img_bgr)
        if embedding is None:
            return False, 0.0

        # 타겟들과 비교
        max_sim = -1.0
        max_sim_type = "none"

        # 1. 메인 임베딩과 비교 (항상 수행)
        for idx, target_emb in enumerate(target_embeddings):
            sim = np.dot(embedding, target_emb)
            if sim > max_sim:
                max_sim = sim
                max_sim_type = f"main_{idx}"

        # 2. Multi-Thumbnail 임베딩과 비교 (옵션)
        if self.use_multi_embedding:
            for target_idx, multi_emb_list in enumerate(multi_embeddings):
                for emb_idx, emb in enumerate(multi_emb_list):
                    emb_array = np.array(emb)
                    # L2 정규화
                    emb_array = emb_array / (np.linalg.norm(emb_array) + 1e-8)
                    sim = np.dot(embedding, emb_array)
                    if sim > max_sim:
                        max_sim = sim
                        max_sim_type = f"multi_{target_idx}_{emb_idx}"

        # 임계값 비교
        is_target = max_sim > self.threshold

        # 상세 로그 (100 프레임마다 또는 threshold 근처)
        if frame_idx % 100 == 0 or (max_sim > self.threshold - 0.1 and max_sim < self.threshold + 0.1):
            logger.info(
                f"  Frame {frame_idx}: similarity={max_sim:.3f} (threshold={self.threshold:.3f}, "
                f"type={max_sim_type}) → {'PRESERVE' if is_target else 'BLUR'}"
            )

        return is_target, max_sim

    def _apply_blur(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        blur_type: str = 'pixelate',
        blur_strength: int = 15,
        padding: int = 20
    ) -> np.ndarray:
        """
        얼굴 영역에 블러 적용

        Args:
            frame: 원본 프레임
            x1, y1, x2, y2: 얼굴 바운딩 박스
            blur_type: 'pixelate' 또는 'gaussian'
            blur_strength: 블러 강도
            padding: 얼굴 주변 패딩

        Returns:
            블러 처리된 프레임
        """
        h, w = frame.shape[:2]

        # 패딩 추가
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if roi.size == 0:
            return frame

        if blur_type == 'pixelate':
            # 픽셀화 (모자이크)
            roi_h, roi_w = roi.shape[:2]
            factor = max(1, max(roi_h, roi_w) // blur_strength)

            if factor > 1:
                small = cv2.resize(roi, (roi_w // factor, roi_h // factor), interpolation=cv2.INTER_NEAREST)
                blurred_roi = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                # 너무 작으면 가우시안 블러 fallback
                k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
                k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
                blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        elif blur_type == 'gaussian':
            # 가우시안 블러
            k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
            k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        else:
            # 기본값: 가우시안 블러
            k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
            k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        frame[y1_pad:y2_pad, x1_pad:x2_pad] = blurred_roi
        return frame

    def process_video(
        self,
        video_path: str,
        output_path: str,
        face_models: List[Dict],
        progress_callback: Optional[callable] = None,
        blur_type: str = 'pixelate',
        blur_strength: int = 15,
        threshold: float = 0.6
    ) -> bool:
        """
        비디오 블러 처리 파이프라인

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            face_models: Face 모델 리스트 [{'id', 'embedding', 'embeddings', 'is_blurred'}, ...]
            progress_callback: 진행률 콜백 함수
            blur_type: 'pixelate' 또는 'gaussian'
            blur_strength: 블러 강도 (높을수록 약함)
            threshold: 유사도 임계값 (기본 0.6)

        Returns:
            성공 여부
        """
        try:
            logger.info("=" * 80)
            logger.info("🎬 Starting Video Blur Processing (Single-Pass)")
            logger.info("=" * 80)

            # threshold 업데이트
            self.threshold = threshold

            # 1. 타겟 임베딩 준비 (is_blurred=False인 얼굴들 - 보존할 얼굴들)
            target_embeddings = []
            multi_embeddings = []

            logger.info(f"📋 Total face_models: {len(face_models)}")

            for fm in face_models:
                is_blurred_val = fm.get('is_blurred', True)
                logger.info(f"Face {fm.get('id')}: is_blurred={is_blurred_val}")

                if not is_blurred_val:  # is_blurred=False (보존할 타겟)
                    # 메인 임베딩
                    emb = np.array(fm['embedding'])
                    emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2 정규화
                    target_embeddings.append(emb)

                    # Multi-Thumbnail 임베딩
                    if 'embeddings' in fm and fm['embeddings']:
                        multi_embeddings.append(fm['embeddings'])

                    logger.info(f"  → Added as TARGET (will be PRESERVED - NO blur)")

            logger.info(f"🎯 Targets to preserve: {len(target_embeddings)}")
            if target_embeddings:
                logger.info("✅ Selected faces will be preserved (not blurred)")
            else:
                logger.warning("⚠️  No target faces selected! ALL faces will be blurred!")

            # 2. 비디오 파일 열기
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            logger.info(f"📹 Video: {width}x{height}, {fps} fps, {total_frames} frames")

            # 3. VideoWriter 생성
            temp_output = output_path + ".temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {temp_output}")

            # 4. 프레임별 처리
            frame_idx = 0
            blur_count = 0
            skip_count = 0
            preserved_count = 0

            logger.info("🎞️  Processing frames...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO 얼굴 감지
                results = self.yolo_model(frame, conf=0.4, verbose=False, device=self.device)

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # 좌표 보정
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        # 너무 작은 얼굴은 스킵
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            skip_count += 1
                            continue

                        # 얼굴 영역 추출
                        face_img_bgr = frame[y1:y2, x1:x2].copy()

                        # ⭐ 기본값: 블러 적용
                        should_blur = True

                        # 타겟과 비교 (타겟이 있는 경우만)
                        if target_embeddings or multi_embeddings:
                            is_target, similarity = self._is_target_face(
                                face_img_bgr,
                                target_embeddings,
                                multi_embeddings,
                                frame_idx
                            )

                            if is_target:
                                # 타겟 얼굴 → 블러 해제
                                should_blur = False
                                preserved_count += 1

                        # 블러 적용
                        if should_blur:
                            frame = self._apply_blur(
                                frame, x1, y1, x2, y2,
                                blur_type=blur_type,
                                blur_strength=blur_strength
                            )
                            blur_count += 1

                # 프레임 저장
                out.write(frame)

                # 진행률 업데이트
                frame_idx += 1
                if progress_callback and frame_idx % 30 == 0:
                    pct = int((frame_idx / total_frames) * 90)
                    pct = min(pct, 90)
                    progress_callback(pct)

                    if frame_idx % 300 == 0:
                        logger.info(
                            f"📊 Processed {frame_idx}/{total_frames} frames | "
                            f"Blurred: {blur_count} | Preserved: {preserved_count} | Skipped: {skip_count}"
                        )

                # 메모리 정리
                if frame_idx % 500 == 0:
                    gc.collect()

            cap.release()
            out.release()

            logger.info(f"✅ Processing completed: {frame_idx} frames")
            logger.info(f"📊 Blurred: {blur_count}, Preserved: {preserved_count}, Skipped: {skip_count}")

            # 5. H.264 인코딩
            logger.info("🎞️  Encoding to H.264...")
            self._encode_h264(temp_output, output_path)

            if progress_callback:
                progress_callback(100)

            logger.info("=" * 80)
            logger.info("✅ Video processing completed successfully!")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}", exc_info=True)
            return False

    def _encode_h264(self, input_path: str, output_path: str):
        """FFmpeg로 H.264 인코딩"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]

            logger.info(f"🎬 Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3600  # 1시간 타임아웃
            )

            # 임시 파일 삭제
            if os.path.exists(input_path):
                os.remove(input_path)

            logger.info("✅ FFmpeg encoding completed")

        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg encoding timeout")
            # Fallback: 임시 파일을 그대로 사용
            if os.path.exists(input_path) and not os.path.exists(output_path):
                shutil.move(input_path, output_path)
                logger.warning("⚠️  Using temp file as output (FFmpeg timeout)")

        except Exception as e:
            logger.error(f"❌ FFmpeg encoding failed: {e}")
            # Fallback: 임시 파일을 그대로 사용
            if os.path.exists(input_path) and not os.path.exists(output_path):
                shutil.move(input_path, output_path)
                logger.warning("⚠️  Using temp file as output (FFmpeg failed)")
