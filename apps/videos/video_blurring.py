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
import os
import shutil
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

logger = logging.getLogger(__name__)


from .adaface_wrapper import AdaFaceWrapper

class VideoBlurrer:
    """
    비디오 블러 처리 클래스 (Two-Pass Architecture)
    
    Pass 1: 분석 (Tracking & Identification)
    - 전체 프레임을 스캔하여 얼굴 궤적(Trajectory) 수집
    - Tracklet Averaging을 통해 각 트랙의 신원(블러 여부) 결정
    
    Refinement: 보정
    - Stitching: 끊긴 트랙 연결
    - Interpolation: 끊긴 구간 보간
    - Smoothing: 궤적 스무딩 (Moving Average)
    
    Pass 2: 렌더링
    - 보정된 궤적을 기반으로 사각형(Rectangle) 블러 적용
    - Padding 적용
    - H.264 인코딩
    """
    
    def __init__(self, yolo_model_path: str, device: str = 'auto', threshold: float = 0.95):
        self.device = device
        self.threshold = threshold
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logger.info(f"VideoBlurrer initialized with device: {self.device}, threshold: {self.threshold}")

        # 1. YOLO 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        
        # 2. AdaFace 모델 로드 (매칭용)
        try:
            # 절대 경로 계산
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # ViT 모델 사용 (사용자가 model.pt를 다운로드하여 이 이름으로 저장했다고 가정)
            weight_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.pt')
            
            # 만약 ViT 파일이 없으면 IR50으로 폴백 (안전장치)
            if not os.path.exists(weight_path):
                # 혹시 ckpt로 저장했을 수도 있으니 확인
                ckpt_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.ckpt')
                if os.path.exists(ckpt_path):
                    weight_path = ckpt_path
                else:
                    logger.warning(f"ViT weights not found at {weight_path}, falling back to IR-50")
                    weight_path = os.path.join(base_dir, 'weights', 'adaface_ir50_ms1mv2.ckpt')
            
            self.adaface_model = AdaFaceWrapper(weight_path, device=self.device)
            logger.info(f"AdaFace model loaded from {weight_path}")
        except Exception as e:
            logger.error(f"Failed to load AdaFace: {e}")
            self.adaface_model = None

    def _get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 이미지에서 임베딩 추출 (AdaFace)"""
        if self.adaface_model is None:
            return None
        return self.adaface_model.get_embedding(face_img)

    def _match_face(self, current_embedding: np.ndarray, face_models: List[Dict], threshold: float = 0.9) -> Optional[Dict]:
        """현재 얼굴 임베딩과 DB 모델 비교"""
        if current_embedding is None or not face_models:
            return None
            
        best_match = None
        max_similarity = -1.0
        
        for face_model in face_models:
            model_embedding = np.array(face_model['embedding'])
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

    def _analyze_video(self, video_path: str, face_models: List[Dict], progress_callback: Optional[callable] = None) -> Tuple[Dict, Dict, Dict]:
        """
        Pass 1: 비디오 분석
        - 트래킹 수행 및 궤적 수집
        - Tracklet Averaging으로 블러 여부 결정
        """
        logger.info("Pass 1: Analyzing video for tracking and identification...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 궤적 저장소: track_id -> list of (frame_idx, x1, y1, x2, y2)
        raw_tracks = {}
        
        # 임베딩 저장소: track_id -> list of embeddings
        track_embeddings = {}
        
        # 최종 결정: track_id -> is_blurred (bool)
        track_decisions = {}
        
        frame_idx = 0
        
        # YOLO Tracking
        results = self.yolo_model.track(
            source=video_path,
            conf=0.4,
            persist=True,
            verbose=False,
            stream=True,
            device=self.device,
            half=(self.device == 'cuda'),
            imgsz=640,
            tracker="botsort.yaml"
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
                    
                    # 1. 궤적 저장
                    if track_id not in raw_tracks:
                        raw_tracks[track_id] = []
                    raw_tracks[track_id].append((frame_idx, x1, y1, x2, y2))
                    
                    # 2. 임베딩 수집 (Tracklet Averaging을 위해)
                    # 얼굴 이미지가 너무 작으면 스킵 (30x30 미만)
                    if (x2 - x1) > 30 and (y2 - y1) > 30:
                        face_img = frame[y1:y2, x1:x2]
                        embedding = self._get_embedding(face_img)
                        
                        if embedding is not None:
                            if track_id not in track_embeddings:
                                track_embeddings[track_id] = []
                            track_embeddings[track_id].append(embedding)
            
            frame_idx += 1
            if progress_callback and frame_idx % 100 == 0:
                # Pass 1은 전체 진행의 40% 차지
                pct = int((frame_idx / total_frames) * 40)
                progress_callback(pct)
                
        cap.release()
        
        # 3. Tracklet Averaging & Decision
        logger.info(f"Analyzing {len(raw_tracks)} tracks with Tracklet Averaging...")
        
        for track_id in raw_tracks.keys():
            embeddings = track_embeddings.get(track_id, [])
            
            is_blurred = True # 기본값: Unknown -> Blur
            face_id = None
            
            if len(embeddings) > 0:
                # 평균 임베딩 계산
                # (N, 512) or (N, 768) -> (512,) or (768,)
                avg_embedding = np.mean(embeddings, axis=0)
                
                # 정규화 (Unit Vector)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                    
                    # 매칭 수행
                    matched = self._match_face(avg_embedding, face_models, self.threshold)
                    if matched:
                        face_id = matched['id']
                        is_blurred = matched['is_blurred']
                        logger.debug(f"Track {track_id} matched to Face {face_id} (Blur: {is_blurred})")
                    else:
                        logger.debug(f"Track {track_id} - No match found (Blur: True)")
            
            track_decisions[track_id] = {
                'is_blurred': is_blurred,
                'face_id': face_id
            }
        
        logger.info(f"Pass 1 completed. Collected {len(raw_tracks)} tracks.")
        return raw_tracks, track_decisions, {'width': width, 'height': height, 'fps': cap.get(cv2.CAP_PROP_FPS), 'total_frames': total_frames}

    def _stitch_tracks(self, raw_tracks: Dict, track_decisions: Dict, max_gap: int = 30) -> Tuple[Dict, Dict]:
        """
        같은 사람(face_id)으로 식별된 트랙들을 연결 (Stitching)
        - 시간적으로 가까운 트랙들을 하나로 병합
        - 중간에 빈 구간은 보간(Interpolation)
        """
        logger.info("Stitching fragmented tracks...")
        
        # 1. Face ID별로 트랙 그룹화
        # face_id -> list of track_ids
        face_groups = {}
        for tid, decision in track_decisions.items():
            face_id = decision['face_id']
            if face_id is not None:
                if face_id not in face_groups:
                    face_groups[face_id] = []
                face_groups[face_id].append(tid)
        
        stitched_tracks = raw_tracks.copy()
        updated_decisions = track_decisions.copy()
        
        # 2. 각 그룹별로 병합 시도
        for face_id, track_ids in face_groups.items():
            if len(track_ids) < 2:
                continue
                
            # 시작 프레임 순으로 정렬
            # track_id -> (start_frame, end_frame, points)
            track_info = []
            for tid in track_ids:
                points = raw_tracks[tid]
                points.sort(key=lambda x: x[0])
                track_info.append({
                    'tid': tid,
                    'start': points[0][0],
                    'end': points[-1][0],
                    'points': points
                })
            
            track_info.sort(key=lambda x: x['start'])
            
            # 병합 로직
            merged_tracks = []
            current_track = track_info[0]
            
            for i in range(1, len(track_info)):
                next_track = track_info[i]
                
                # 시간 차이 계산
                gap = next_track['start'] - current_track['end']
                
                if 0 < gap < max_gap:
                    # 병합 가능: 두 트랙을 잇고 중간을 보간
                    logger.info(f"Stitching tracks {current_track['tid']} and {next_track['tid']} (Gap: {gap})")
                    
                    # 보간 포인트 생성
                    last_p = current_track['points'][-1]
                    first_p = next_track['points'][0]
                    
                    interpolated = []
                    for step in range(1, gap):
                        ratio = step / gap
                        frame_idx = last_p[0] + step
                        x1 = int(last_p[1] + (first_p[1] - last_p[1]) * ratio)
                        y1 = int(last_p[2] + (first_p[2] - last_p[2]) * ratio)
                        x2 = int(last_p[3] + (first_p[3] - last_p[3]) * ratio)
                        y2 = int(last_p[4] + (first_p[4] - last_p[4]) * ratio)
                        interpolated.append((frame_idx, x1, y1, x2, y2))
                    
                    # 포인트 합치기
                    current_track['points'].extend(interpolated)
                    current_track['points'].extend(next_track['points'])
                    current_track['end'] = next_track['end']
                    
                    # 병합된 트랙(next_track)은 삭제 대상
                    if next_track['tid'] in stitched_tracks:
                        del stitched_tracks[next_track['tid']]
                    if next_track['tid'] in updated_decisions:
                        del updated_decisions[next_track['tid']]
                        
                else:
                    # 병합 불가: 현재 트랙 완료, 다음 트랙으로 이동
                    current_track = next_track
                    
        return stitched_tracks, updated_decisions

    def _stitch_tracks(self, raw_tracks: Dict, track_decisions: Dict, max_gap: int = 60) -> Tuple[Dict, Dict]:
        """
        같은 사람(face_id)으로 식별된 트랙들을 연결 (Stitching)
        - 시간적으로 가까운 트랙들을 하나로 병합
        - 중간에 빈 구간은 보간(Interpolation)
        
        [Identity Propagation & Side Profile Robustness]
        - Unknown 트랙이 Known 트랙과 시공간적으로 연결되면 ID 전파
        - Soft Matching: 갭 60프레임(2초), 거리 100픽셀 이내면 "같은 사람"으로 간주 (옆모습 보정)
        """
        logger.info("Stitching fragmented tracks (Side Profile Robustness Enabled)...")
        
        # 1. Face ID별로 트랙 그룹화
        # face_id -> list of track_ids
        face_groups = {}
        unknown_tracks = []
        
        for tid, decision in track_decisions.items():
            face_id = decision['face_id']
            if face_id is not None:
                if face_id not in face_groups:
                    face_groups[face_id] = []
                face_groups[face_id].append(tid)
            else:
                unknown_tracks.append(tid)
        
        stitched_tracks = raw_tracks.copy()
        updated_decisions = track_decisions.copy()
        
        # 2. Known 트랙끼리 병합 (기존 로직)
        for face_id, track_ids in face_groups.items():
            if len(track_ids) < 2:
                continue
                
            # 시작 프레임 순으로 정렬
            track_info = []
            for tid in track_ids:
                points = raw_tracks[tid]
                points.sort(key=lambda x: x[0])
                track_info.append({
                    'tid': tid,
                    'start': points[0][0],
                    'end': points[-1][0],
                    'points': points
                })
            
            track_info.sort(key=lambda x: x['start'])
            
            # 병합 로직
            current_track = track_info[0]
            
            for i in range(1, len(track_info)):
                next_track = track_info[i]
                gap = next_track['start'] - current_track['end']
                
                if 0 < gap < max_gap:
                    # 병합 가능
                    logger.info(f"Stitching Known tracks {current_track['tid']} and {next_track['tid']} (Gap: {gap})")
                    
                    # 보간 포인트 생성
                    last_p = current_track['points'][-1]
                    first_p = next_track['points'][0]
                    
                    interpolated = []
                    for step in range(1, gap):
                        ratio = step / gap
                        frame_idx = last_p[0] + step
                        x1 = int(last_p[1] + (first_p[1] - last_p[1]) * ratio)
                        y1 = int(last_p[2] + (first_p[2] - last_p[2]) * ratio)
                        x2 = int(last_p[3] + (first_p[3] - last_p[3]) * ratio)
                        y2 = int(last_p[4] + (first_p[4] - last_p[4]) * ratio)
                        interpolated.append((frame_idx, x1, y1, x2, y2))
                    
                    current_track['points'].extend(interpolated)
                    current_track['points'].extend(next_track['points'])
                    current_track['end'] = next_track['end']
                    
                    if next_track['tid'] in stitched_tracks:
                        del stitched_tracks[next_track['tid']]
                    if next_track['tid'] in updated_decisions:
                        del updated_decisions[next_track['tid']]
                else:
                    current_track = next_track

        # 3. Identity Propagation (Unknown -> Known 병합)
        # Known 트랙들을 다시 정리 (병합된 결과 반영)
        known_tracks = []
        for tid, decision in updated_decisions.items():
            if decision['face_id'] is not None:
                points = stitched_tracks[tid]
                points.sort(key=lambda x: x[0])
                known_tracks.append({
                    'tid': tid,
                    'face_id': decision['face_id'],
                    'is_blurred': decision['is_blurred'],
                    'start': points[0][0],
                    'end': points[-1][0],
                    'start_point': points[0],
                    'end_point': points[-1],
                    'points': points
                })
        
        # Unknown 트랙 처리
        for unknown_tid in unknown_tracks:
            if unknown_tid not in stitched_tracks: continue
            
            u_points = stitched_tracks[unknown_tid]
            u_points.sort(key=lambda x: x[0])
            u_start = u_points[0][0]
            u_end = u_points[-1][0]
            u_start_p = u_points[0]
            u_end_p = u_points[-1]
            
            best_match = None
            min_dist = float('inf')
            
            # Soft Matching Parameters
            # Gap: 60 frames (2s)
            # Dist: 100 pixels
            
            for k_track in known_tracks:
                # Case A: Known -> Unknown (Known이 앞에 있음)
                gap_a = u_start - k_track['end']
                if 0 < gap_a < max_gap:
                    # 거리 계산 (Known 끝점 vs Unknown 시작점)
                    dist = np.sqrt((k_track['end_point'][1] - u_start_p[1])**2 + (k_track['end_point'][2] - u_start_p[2])**2)
                    if dist < 100: # 100픽셀 이내 (Soft Matching)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = (k_track, 'append')
                            
                # Case B: Unknown -> Known (Unknown이 앞에 있음)
                gap_b = k_track['start'] - u_end
                if 0 < gap_b < max_gap:
                    # 거리 계산 (Unknown 끝점 vs Known 시작점)
                    dist = np.sqrt((u_end_p[1] - k_track['start_point'][1])**2 + (u_end_p[2] - k_track['start_point'][2])**2)
                    if dist < 100: # 100픽셀 이내 (Soft Matching)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = (k_track, 'prepend')
            
            if best_match:
                target_track, mode = best_match
                logger.info(f"Propagating Identity {target_track['face_id']} to Unknown Track {unknown_tid} ({mode}, dist={min_dist:.1f}) - Side Profile Fix")
                
                # 병합 수행
                target_tid = target_track['tid']
                
                if mode == 'append':
                    # 보간
                    last_p = target_track['points'][-1]
                    first_p = u_points[0]
                    gap = u_start - target_track['end']
                    
                    interpolated = []
                    for step in range(1, gap):
                        ratio = step / gap
                        frame_idx = last_p[0] + step
                        x1 = int(last_p[1] + (first_p[1] - last_p[1]) * ratio)
                        y1 = int(last_p[2] + (first_p[2] - last_p[2]) * ratio)
                        x2 = int(last_p[3] + (first_p[3] - last_p[3]) * ratio)
                        y2 = int(last_p[4] + (first_p[4] - last_p[4]) * ratio)
                        interpolated.append((frame_idx, x1, y1, x2, y2))
                        
                    stitched_tracks[target_tid].extend(interpolated)
                    stitched_tracks[target_tid].extend(u_points)
                    
                    # 메타데이터 업데이트
                    target_track['end'] = u_end
                    target_track['end_point'] = u_end_p
                    target_track['points'] = stitched_tracks[target_tid] # 참조 갱신
                    
                else: # prepend
                    # 보간
                    last_p = u_points[-1]
                    first_p = target_track['points'][0]
                    gap = target_track['start'] - u_end
                    
                    interpolated = []
                    for step in range(1, gap):
                        ratio = step / gap
                        frame_idx = last_p[0] + step
                        x1 = int(last_p[1] + (first_p[1] - last_p[1]) * ratio)
                        y1 = int(last_p[2] + (first_p[2] - last_p[2]) * ratio)
                        x2 = int(last_p[3] + (first_p[3] - last_p[3]) * ratio)
                        y2 = int(last_p[4] + (first_p[4] - last_p[4]) * ratio)
                        interpolated.append((frame_idx, x1, y1, x2, y2))
                        
                    # 앞에 붙이기
                    new_points = u_points + interpolated + stitched_tracks[target_tid]
                    stitched_tracks[target_tid] = new_points
                    
                    # 메타데이터 업데이트
                    target_track['start'] = u_start
                    target_track['start_point'] = u_start_p
                    target_track['points'] = new_points
                
                # Unknown 트랙 삭제
                if unknown_tid in stitched_tracks:
                    del stitched_tracks[unknown_tid]
                if unknown_tid in updated_decisions:
                    del updated_decisions[unknown_tid]
                        
        return stitched_tracks, updated_decisions

    def _refine_trajectories(self, raw_tracks: Dict, track_decisions: Dict) -> Dict:
        """
        Refinement: 궤적 보정
        1. Stitching: 끊긴 트랙 연결
        2. Interpolation: 트랙 내 공백 보간
        3. Smoothing: 이동 평균 적용
        """
        # 1. Stitching
        stitched_tracks, _ = self._stitch_tracks(raw_tracks, track_decisions)
        
        logger.info("Refining trajectories (Interpolation & Smoothing)...")
        refined_tracks = {}
        
        for track_id, points in stitched_tracks.items():
            # points: list of (frame_idx, x1, y1, x2, y2)
            points.sort(key=lambda x: x[0])
            
            # 2. Interpolation (트랙 내부의 작은 구멍 메우기)
            interpolated = []
            for i in range(len(points) - 1):
                curr = points[i]
                next_p = points[i+1]
                
                interpolated.append(curr)
                
                gap = next_p[0] - curr[0]
                if 1 < gap < 10: # 10프레임 미만 공백만 보간
                    for step in range(1, gap):
                        ratio = step / gap
                        frame_idx = curr[0] + step
                        x1 = int(curr[1] + (next_p[1] - curr[1]) * ratio)
                        y1 = int(curr[2] + (next_p[2] - curr[2]) * ratio)
                        x2 = int(curr[3] + (next_p[3] - curr[3]) * ratio)
                        y2 = int(curr[4] + (next_p[4] - curr[4]) * ratio)
                        interpolated.append((frame_idx, x1, y1, x2, y2))
            
            interpolated.append(points[-1])
            
            # 3. Smoothing (Moving Average)
            window_size = 5
            smoothed = []
            
            # 좌표별로 분리
            frames = [p[0] for p in interpolated]
            coords = np.array([p[1:] for p in interpolated]) # [[x1, y1, x2, y2], ...]
            
            # 이동 평균 적용
            if len(coords) >= window_size:
                # 커널 생성
                kernel = np.ones(window_size) / window_size
                
                # 각 좌표(x1, y1, x2, y2)에 대해 컨볼루션
                smooth_coords = np.zeros_like(coords)
                for i in range(4):
                    # same 모드로 패딩 처리하여 길이 유지
                    smooth_coords[:, i] = np.convolve(coords[:, i], kernel, mode='same')
                
                # 정수 변환
                smooth_coords = smooth_coords.astype(int)
                
                # 다시 합치기
                for f, c in zip(frames, smooth_coords):
                    smoothed.append((f, *c))
            else:
                smoothed = interpolated
                
            refined_tracks[track_id] = smoothed
            
        return refined_tracks

    def _render_video(self, video_path: str, output_path: str, refined_tracks: Dict, track_decisions: Dict, meta: Dict, progress_callback: Optional[callable] = None) -> bool:
        """
        Pass 2: 렌더링
        - 보정된 궤적을 사용하여 타원형 블러 적용
        - Padding 적용으로 블러 영역 확장
        """
        logger.info("Pass 2: Rendering video with ellipse blur...")
        
        cap = cv2.VideoCapture(video_path)
        width = meta['width']
        height = meta['height']
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = meta['total_frames']
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 빠른 조회를 위해 프레임별 트랙 정보 재구성
        # frame_idx -> list of (track_id, x1, y1, x2, y2)
        frame_map = {}
        for tid, points in refined_tracks.items():
            # Decision 확인 (dict or bool)
            decision = track_decisions.get(tid, {'is_blurred': True})
            if isinstance(decision, dict):
                should_blur = decision.get('is_blurred', True)
            else:
                should_blur = decision
                
            if not should_blur:
                continue
                
            for p in points:
                f_idx, x1, y1, x2, y2 = p
                if f_idx not in frame_map:
                    frame_map[f_idx] = []
                frame_map[f_idx].append((tid, x1, y1, x2, y2))
        
        frame_idx = 0
        padding = 20 # 블러 영역 확장 (픽셀)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in frame_map:
                for _, x1, y1, x2, y2 in frame_map[frame_idx]:
                    # Padding 적용
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)
                    
                    # 사각형 블러 적용
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        # 가우시안 블러 적용
                        # 커널 크기는 ROI 크기에 비례하게 설정
                        k_w = (x2 - x1) // 3 | 1  # 홀수여야 함
                        k_h = (y2 - y1) // 3 | 1
                        blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
                        frame[y1:y2, x1:x2] = blurred_roi
            
            out.write(frame)
            
            frame_idx += 1
            if progress_callback and frame_idx % 100 == 0:
                # Pass 2는 40% ~ 90% 구간
                pct = 40 + int((frame_idx / total_frames) * 50)
                progress_callback(pct)
                
        cap.release()
        out.release()
        return True

    def process_video(self, video_path: str, output_path: str, face_models: List[Dict], progress_callback: Optional[callable] = None) -> bool:
        """전체 파이프라인 실행"""
        try:
            # 1. Pass 1: 분석
            raw_tracks, track_decisions, meta = self._analyze_video(video_path, face_models, progress_callback)
            
            # 2. Refinement: 보정 (Stitching 포함)
            refined_tracks = self._refine_trajectories(raw_tracks, track_decisions)
            
            # 3. Pass 2: 렌더링
            self._render_video(video_path, output_path, refined_tracks, track_decisions, meta, progress_callback)
            
            # 4. Encoding: H.264 변환
            self._encode_h264(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            return False

    def _encode_h264(self, output_path: str):
        """FFmpeg로 H.264 인코딩"""
        try:
            import subprocess
            import shutil
            
            temp_output = output_path + ".temp.mp4"
            if os.path.exists(output_path):
                shutil.move(output_path, temp_output)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]
            
            logger.info(f"Running FFmpeg: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                shutil.move(temp_output, output_path)
