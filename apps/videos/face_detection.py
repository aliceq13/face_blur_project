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
from .adaface_wrapper import AdaFaceWrapper

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
    메모리 최적화: 이미지 대신 임베딩만 저장
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.embeddings: List[np.ndarray] = []  # 이미지 대신 임베딩만 저장
        self.best_face: Optional[DetectedFace] = None  # 썸네일용 대표 얼굴 1개만
        self.avg_embedding: Optional[np.ndarray] = None
        self.appearance_count: int = 0
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None

    def add_face(self, face: DetectedFace):
        """얼굴 추가 - 임베딩만 저장하고 이미지는 썸네일 선택용으로만 사용"""
        self.appearance_count += 1

        if self.first_frame is None:
            self.first_frame = face.frame_idx
        self.last_frame = face.frame_idx

        # 임베딩이 있으면 저장
        if face.embedding is not None:
            self.embeddings.append(face.embedding)

        # 대표 얼굴 업데이트 (선명도 기준)
        if self.best_face is None or face.clarity > self.best_face.clarity:
            # 기존 best_face 이미지 메모리 해제
            if self.best_face is not None:
                self.best_face.face_img = None
            self.best_face = face
        else:
            # 선택되지 않은 얼굴 이미지 즉시 삭제
            face.face_img = None

    def compute_average_embedding(self):
        """평균 임베딩 계산 후 개별 임베딩 삭제"""
        if not self.embeddings:
            return

        # 평균 계산 및 정규화
        avg_emb = np.mean(self.embeddings, axis=0)
        self.avg_embedding = normalize(avg_emb.reshape(1, -1))[0]

        # 메모리 절약: 개별 임베딩 삭제
        self.embeddings = []


class FaceDetectionPipeline:
    """얼굴 감지 및 인식 파이프라인 (ArcFace + Tracklet Clustering)"""

    # 청크 기반 처리 설정 (성능 우선)
    CHUNK_SIZE = 1500  # 1500 프레임씩 처리 (속도 향상)
    MEMORY_CHECK_INTERVAL = 200  # 200 프레임마다 메모리 체크
    MEMORY_LIMIT_GB = 7.0  # 7GB 초과 시 강제 GC

    def __init__(
        self,
        yolo_model_path: str = None,
        device: str = 'auto',
        sample_rate: int = 1  # Tracking을 위해 기본적으로 모든 프레임 처리 권장
    ):
        self.sample_rate = sample_rate
        self.frames_processed = 0

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

        # 2. InsightFace (Detection용)
        # 3. AdaFace (Recognition용)
        try:
            self.arcface_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.arcface_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            
            # AdaFace 로드
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # ViT 모델 사용
            weight_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.pt')
            
            # 만약 ViT 파일이 없으면 IR50으로 폴백
            if not os.path.exists(weight_path):
                # 혹시 ckpt로 저장했을 수도 있으니 확인
                ckpt_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.ckpt')
                if os.path.exists(ckpt_path):
                    weight_path = ckpt_path
                else:
                    logger.warning(f"ViT weights not found at {weight_path}, falling back to IR-50")
                    weight_path = os.path.join(base_dir, 'weights', 'adaface_ir50_ms1mv2.ckpt')
                
            self.adaface_model = AdaFaceWrapper(weight_path, device=self.device)
            
            logger.info(f"Models loaded: YOLO, InsightFace(Det), AdaFace(Rec) - {os.path.basename(weight_path)}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            self.arcface_app = None
            self.adaface_model = None

    def _calculate_clarity(self, img: np.ndarray) -> float:
        """이미지 선명도 계산 (Laplacian Variance)"""
        try:
            if img.size == 0:
                return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0

    def _check_memory_usage(self):
        """메모리 사용량 체크 및 자동 정리"""
        import psutil
        import gc

        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)

        if memory_gb > self.MEMORY_LIMIT_GB:
            logger.warning(f"Memory usage high: {memory_gb:.2f} GB, forcing garbage collection")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return memory_gb

    def _process_chunk(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        tracklets: Dict[int, Tracklet],
        conf_threshold: float = 0.5
    ) -> tuple:
        """
        청크 단위로 프레임 처리 (메모리 효율적)

        Args:
            video_path: 비디오 경로
            start_frame: 시작 프레임 인덱스
            end_frame: 종료 프레임 인덱스
            tracklets: Tracklet 딕셔너리 (누적)
            conf_threshold: YOLO confidence threshold

        Returns:
            (embedding_success, embedding_fail) 튜플
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        embedding_success = 0
        embedding_fail = 0

        # YOLO tracking 실행 (청크 범위만)
        # stream=True로 설정하여 메모리 효율적으로 처리
        results = self.yolo_model.track(
            source=video_path,
            conf=0.4,  # Lower threshold
            persist=True,
            verbose=False,
            stream=True,  # 메모리 효율적 스트리밍
            vid_stride=self.sample_rate,
            device=self.device,
            half=True,  # FP16 사용 (GPU 성능 향상)
            imgsz=640,
            tracker="botsort.yaml",
            batch=16  # 배치 크기 증가 (GPU 활용도 향상)
        )

        # 청크 범위 내 프레임만 처리
        for idx, result in enumerate(results):
            frame_idx = start_frame + (idx * self.sample_rate)
            if frame_idx >= end_frame:
                break

            frame = result.orig_img

            if not result.boxes:
                continue

            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # 좌표 보정
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # 얼굴 크롭
                face_img_bgr = frame[y1:y2, x1:x2]
                face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
                clarity = self._calculate_clarity(face_img_rgb)

                detected_face = DetectedFace(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    face_img=face_img_rgb,
                    track_id=track_id,
                    clarity=clarity
                )

                # 임베딩 추출 (AdaFace)
                if self.adaface_model:
                    embedding = self.adaface_model.get_embedding(face_img_bgr)
                    if embedding is not None:
                        detected_face.embedding = embedding
                        embedding_success += 1
                    else:
                        embedding_fail += 1
                
                # Tracklet에 추가
                if track_id not in tracklets:
                    tracklets[track_id] = Tracklet(track_id)
                tracklets[track_id].add_face(detected_face)

            # 메모리 체크
            self.frames_processed += 1
            if self.frames_processed % self.MEMORY_CHECK_INTERVAL == 0:
                self._check_memory_usage()

        cap.release()
        return embedding_success, embedding_fail

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        conf_threshold: float = 0.5,
        sim_threshold: float = 0.6,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        전체 파이프라인 실행: Tracking -> Embedding -> Tracklet Clustering
        청크 기반 처리로 메모리 효율 최적화
        """
        logger.info(f"Starting advanced face analysis for {video_path}")

        # 비디오 정보 읽기
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logger.info(f"Total frames: {total_frames}, processing in chunks of {self.CHUNK_SIZE}")

        # 청크 계산
        num_chunks = (total_frames + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        tracklets: Dict[int, Tracklet] = {}

        total_embedding_success = 0
        total_embedding_fail = 0

        # 청크별 처리
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * self.CHUNK_SIZE
            end_frame = min((chunk_idx + 1) * self.CHUNK_SIZE, total_frames)

            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame})")

            # 청크 처리
            success, fail = self._process_chunk(
                video_path, start_frame, end_frame, tracklets, conf_threshold
            )

            total_embedding_success += success
            total_embedding_fail += fail

            # 진행률 콜백
            if progress_callback:
                progress_pct = int((end_frame / total_frames) * 100)
                progress_callback(progress_pct, f"청크 {chunk_idx + 1}/{num_chunks} 처리 완료")

            # 청크 간 메모리 정리
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Chunk {chunk_idx + 1} completed. Memory cleaned.")

        logger.info(f"Tracking completed. Found {len(tracklets)} tracklets.")
        logger.info(f"Embedding extraction: {total_embedding_success} success, {total_embedding_fail} fail")

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

                # req_c가 None이면 마지막 레벨 사용
                if req_c is None:
                    req_c = len(num_clust) - 1
                    logger.warning(f"FINCH req_c is None, using last level: {req_c}")

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
            # 클러스터 내 최고 선명도 얼굴 선택 (best_face 사용)
            best_face = None
            best_clarity = 0

            for t in cluster_tracklets:
                if t.best_face and t.best_face.clarity > best_clarity:
                    best_face = t.best_face
                    best_clarity = t.best_face.clarity

            if not best_face:
                continue

            # 썸네일 저장
            thumbnail_filename = f"face_{face_index}.jpg"
            thumbnail_path = os.path.join(output_dir, thumbnail_filename)
            self.save_thumbnail(best_face, thumbnail_path)

            # 프레임 정보 수집 (first_frame, last_frame 사용)
            all_first_frames = [t.first_frame for t in cluster_tracklets if t.first_frame is not None]
            all_last_frames = [t.last_frame for t in cluster_tracklets if t.last_frame is not None]
            total_appearances = sum(t.appearance_count for t in cluster_tracklets)

            # 클러스터 평균 임베딩
            cluster_avg_emb = np.mean([t.avg_embedding for t in cluster_tracklets], axis=0)
            cluster_avg_emb = normalize(cluster_avg_emb.reshape(1, -1))[0]

            result_faces.append({
                'face_index': face_index,
                'thumbnail_path': thumbnail_path,
                'embedding': cluster_avg_emb.tolist(),
                'appearance_count': total_appearances,
                'first_frame': min(all_first_frames) if all_first_frames else 0,
                'last_frame': max(all_last_frames) if all_last_frames else 0
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
