# -*- coding: utf-8 -*-
"""
얼굴 감지 및 추적 파이프라인 (Refactored)

이 모듈은 YOLO(Tracking) + ArcFace + FINCH를 사용하여
비디오에서 얼굴을 감지하고 동일 인물을 정교하게 그룹화합니다.

개선 사항:
1. FaceNet → ArcFace (InsightFace): 인식 정확도 대폭 향상
2. Frame-based → Tracklet-based: 추적(Tracking) 정보를 활용하여 안정성 향상
3. DBSCAN → HAC → FINCH: 파라미터 프리 클러스터링으로 성능 및 자동화 개선
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import normalize
import logging
import insightface
from insightface.app import FaceAnalysis

# FINCH: 파라미터 프리 클러스터링 (HAC 대체)
try:
    from finch import FINCH
    FINCH_AVAILABLE = True
except ImportError:
    FINCH_AVAILABLE = False
    from sklearn.cluster import AgglomerativeClustering
    logging.warning("FINCH not available, falling back to AgglomerativeClustering")

logger = logging.getLogger(__name__)


class DetectedFace:
    """감지된 얼굴 정보를 담는 데이터 클래스"""

    def __init__(
        self,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        confidence: float,
        face_img: np.ndarray,
        track_id: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        clarity: float = 0.0
    ):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.confidence = confidence
        self.face_img = face_img
        self.track_id = track_id  # Tracking ID (ByteTrack)
        self.embedding = embedding
        self.clarity = clarity

    @property
    def bbox_area(self) -> int:
        """바운딩 박스 면적 계산"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class Tracklet:
    """
    동일한 Track ID를 가진 얼굴들의 집합 (시간 연속성 보장)
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.faces: List[DetectedFace] = []
        self.avg_embedding: Optional[np.ndarray] = None

    def add_face(self, face: DetectedFace):
        self.faces.append(face)

    def compute_average_embedding(self):
        """Tracklet 내 얼굴들의 평균 임베딩 계산"""
        if not self.faces:
            return
        
        # 임베딩이 있는 얼굴만 필터링
        valid_embeddings = [f.embedding for f in self.faces if f.embedding is not None]
        if not valid_embeddings:
            return

        # 평균 계산 및 정규화
        avg_emb = np.mean(valid_embeddings, axis=0)
        self.avg_embedding = normalize(avg_emb.reshape(1, -1))[0]


class FaceDetectionPipeline:
    """얼굴 감지 및 인식 파이프라인 (ArcFace + Tracklet Clustering)"""

    def __init__(
        self,
        yolo_model_path: str = None,
        device: str = 'auto',
        sample_rate: int = 1  # Tracking을 위해 기본적으로 모든 프레임 처리 권장
    ):
        self.sample_rate = sample_rate

        # YOLO 모델 경로
        if yolo_model_path is None:
            from django.conf import settings
            yolo_model_path = str(settings.YOLO_FACE_MODEL_PATH)

        # Device 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # 1. YOLO 모델 로드 (Detection & Tracking)
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"YOLO model loaded from {yolo_model_path}")

        # 2. InsightFace ArcFace 모델 로드 (Recognition)
        # buffalo_l: 2023년 기준 성능 좋은 기본 모델 (Detection+Recognition 포함)
        # 여기서는 Recognition(ArcFace) 기능만 주로 활용
        try:
            self.arcface_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            # det_size를 작게 설정하여 작은 crop 이미지에서도 검출 가능하게 함
            self.arcface_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(160, 160))
            logger.info("InsightFace (ArcFace) model loaded")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}. Make sure onnxruntime is installed.")
            self.arcface_app = None

    def _calculate_clarity(self, img: np.ndarray) -> float:
        """이미지 선명도 계산 (Laplacian Variance)"""
        try:
            if img.size == 0:
                return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        conf_threshold: float = 0.5,
        sim_threshold: float = 0.6  # 같은 사람으로 판단할 유사도 임계값 (ArcFace는 보통 0.3~0.6)
    ) -> List[Dict]:
        """
        전체 파이프라인 실행: Tracking -> Embedding -> Tracklet Clustering
        """
        logger.info(f"Starting advanced face analysis for {video_path}")

        # 1. Video Tracking (YOLO)
        # track() 메서드는 제너레이터로 사용하여 메모리 효율적으로 처리 가능
        # stream=True로 설정하여 프레임별로 처리
        results = self.yolo_model.track(
            source=video_path,
            conf=conf_threshold,
            persist=True,  # 객체 추적 유지
            verbose=False,
            stream=True,
            vid_stride=self.sample_rate
        )

        # Tracklet 딕셔너리: {track_id: Tracklet}
        tracklets: Dict[int, Tracklet] = {}

        frame_idx = 0
        embedding_success = 0  # 임베딩 추출 성공 카운터
        embedding_fail = 0     # 임베딩 추출 실패 카운터
        
        # 프레임 순회
        for result in results:
            frame = result.orig_img  # 원본 프레임 (BGR)
            
            # 감지된 객체가 없으면 패스
            if not result.boxes:
                frame_idx += self.sample_rate
                continue

            # 각 박스 처리
            for box in result.boxes:
                # Track ID가 없는 경우(추적 실패)는 무시하거나 -1로 처리
                if box.id is None:
                    continue
                
                track_id = int(box.id.item())
                conf = float(box.conf.item())
                
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # 좌표 보정
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                # 얼굴 이미지 크롭 (BGR 상태)
                face_img_bgr = frame[y1:y2, x1:x2]
                face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
                
                # 선명도 계산
                clarity = self._calculate_clarity(face_img_rgb)

                # DetectedFace 객체 생성
                detected_face = DetectedFace(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    face_img=face_img_rgb,
                    track_id=track_id,
                    clarity=clarity
                )

                # 임베딩 추출 (ArcFace)
                # InsightFace는 BGR 이미지를 입력으로 받음 (OpenCV 포맷)
                if self.arcface_app:
                    try:
                        # crop된 이미지를 충분히 크게 resize하여 InsightFace가 검출할 수 있도록 함
                        # InsightFace는 최소 112x112 이상의 얼굴이 필요
                        h, w = face_img_bgr.shape[:2]

                        # 얼굴 이미지가 너무 작으면 확대
                        min_size = 256  # InsightFace가 검출하기 좋은 크기 (160→256)
                        if h < min_size or w < min_size:
                            scale = max(min_size / h, min_size / w)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            face_img_resized = cv2.resize(face_img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            face_img_resized = face_img_bgr

                        # InsightFace로 임베딩 추출
                        faces = self.arcface_app.get(face_img_resized)
                        if faces:
                            # crop된 이미지 내에서 검출된 얼굴의 임베딩 사용
                            detected_face.embedding = faces[0].embedding
                            embedding_success += 1
                        else:
                            embedding_fail += 1
                            logger.debug(f"InsightFace failed to detect face in crop (frame {frame_idx}, size {h}x{w})")
                    except Exception as e:
                        embedding_fail += 1
                        logger.debug(f"Embedding extraction failed (frame {frame_idx}): {e}")

                # Tracklet에 추가
                if track_id not in tracklets:
                    tracklets[track_id] = Tracklet(track_id)
                
                tracklets[track_id].add_face(detected_face)

            frame_idx += self.sample_rate

        logger.info(f"Tracking completed. Found {len(tracklets)} tracklets.")
        logger.info(f"Embedding extraction: {embedding_success} success, {embedding_fail} fail")

        # 2. Tracklet Clustering (HAC)
        # 각 Tracklet의 평균 임베딩 계산
        valid_tracklets = []
        embeddings = []

        for t in tracklets.values():
            t.compute_average_embedding()
            if t.avg_embedding is not None:
                valid_tracklets.append(t)
                embeddings.append(t.avg_embedding)

        if not valid_tracklets:
            logger.warning("No valid tracklets with embeddings found.")
            return []

        # 클러스터링 실행
        # 1개의 tracklet만 있는 경우 클러스터링 불필요
        if len(valid_tracklets) == 1:
            logger.info("Only 1 tracklet found, skipping clustering.")
            labels = [0]  # 단일 클러스터로 처리
        else:
            # FINCH 사용 가능 시 FINCH로 클러스터링 (파라미터 프리)
            if FINCH_AVAILABLE:
                logger.info("Using FINCH clustering (parameter-free)")
                embeddings_array = np.array(embeddings)

                # FINCH 실행: 완전 자동, 파라미터 조정 불필요
                # c: 각 계층의 클러스터 레이블 (n_samples, n_partitions)
                # num_clust: 각 계층의 클러스터 수
                # req_c: 추천되는 파티션 레벨
                c, num_clust, req_c = FINCH(embeddings_array, distance='cosine', verbose=False)

                # 추천 레벨 사용 (자동으로 최적의 클러스터 수 결정)
                labels = c[:, req_c]

                logger.info(f"FINCH clustering completed: {num_clust[req_c]} unique persons found (level {req_c})")
            else:
                # Fallback: HAC (기존 방식)
                logger.info("Using AgglomerativeClustering (HAC) as fallback")
                # distance_threshold: 같은 사람으로 볼 거리 (코사인 거리 기준)
                # ArcFace의 경우 1 - cosine_similarity가 거리.
                # 보통 threshold 0.4~0.6 정도가 적당 (엄격하게 하려면 낮게)
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1.0 - sim_threshold,  # 거리 임계값
                    metric='cosine',
                    linkage='average'  # 평균 연결법 (안정적)
                )
                labels = clustering.fit_predict(embeddings)
        
        # 클러스터별 그룹화
        clusters: Dict[int, List[Tracklet]] = {}
        for tracklet, label in zip(valid_tracklets, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tracklet)

        logger.info(f"Clustering completed. Found {len(clusters)} unique persons from {len(valid_tracklets)} tracklets.")

        # 3. 결과 생성 및 썸네일 저장
        result_faces = []
        face_index = 1

        for cluster_id, cluster_tracklets in clusters.items():
            # 클러스터 내 모든 얼굴 수집
            all_faces_in_cluster = []
            for t in cluster_tracklets:
                all_faces_in_cluster.extend(t.faces)

            if not all_faces_in_cluster:
                continue

            # 대표 썸네일 선택 (선명도 * 크기)
            best_face = self.select_best_thumbnail(all_faces_in_cluster)

            # 썸네일 저장
            thumbnail_filename = f"face_{face_index}.jpg"
            thumbnail_path = os.path.join(output_dir, thumbnail_filename)
            self.save_thumbnail(best_face, thumbnail_path)

            # 프레임 정보
            frame_indices = [f.frame_idx for f in all_faces_in_cluster]
            
            # 대표 임베딩 (클러스터 내 모든 Tracklet 평균의 평균)
            cluster_avg_emb = np.mean([t.avg_embedding for t in cluster_tracklets], axis=0)
            cluster_avg_emb = normalize(cluster_avg_emb.reshape(1, -1))[0]

            result_faces.append({
                'face_index': face_index,
                'thumbnail_path': thumbnail_path,
                'embedding': cluster_avg_emb.tolist(),
                'appearance_count': len(all_faces_in_cluster),
                'first_frame': min(frame_indices),
                'last_frame': max(frame_indices)
            })
            
            face_index += 1

        return result_faces

    def select_best_thumbnail(self, faces: List[DetectedFace]) -> DetectedFace:
        """최적의 썸네일 선택 (선명도 * sqrt(면적))"""
        def calculate_score(face: DetectedFace) -> float:
            size_factor = np.sqrt(face.bbox_area)
            return face.clarity * size_factor

        sorted_faces = sorted(faces, key=calculate_score, reverse=True)
        return sorted_faces[0]

    def save_thumbnail(self, face: DetectedFace, output_path: str, size=(160, 160)) -> str:
        """썸네일 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized = cv2.resize(face.face_img, size, interpolation=cv2.INTER_AREA)
        bgr_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_img)
        return output_path
