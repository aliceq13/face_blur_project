# -*- coding: utf-8 -*-
"""
얼굴 감지 및 추적 파이프라인

이 모듈은 YOLO Face + FaceNet + DBSCAN을 사용하여
비디오에서 얼굴을 감지하고 동일 인물을 그룹화하는 기능을 제공합니다.

처리 흐름:
1. 비디오 프레임 샘플링 (5프레임마다)
2. YOLO로 얼굴 감지 (bounding box)
3. FaceNet으로 얼굴 임베딩 추출 (512차원 벡터)
4. DBSCAN으로 클러스터링 (동일 인물 그룹화)
5. 대표 썸네일 선택 및 저장
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)


class DetectedFace:
    """감지된 얼굴 정보를 담는 데이터 클래스"""

    def __init__(
        self,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        confidence: float,
        face_img: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        clarity: float = 0.0
    ):
        self.frame_idx = frame_idx
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.face_img = face_img
        self.embedding = embedding
        self.clarity = clarity

    @property
    def bbox_area(self) -> int:
        """바운딩 박스 면적 계산 (썸네일 품질 선택용)"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class FaceDetectionPipeline:
    """얼굴 감지 파이프라인 클래스"""

    def __init__(
        self,
        yolo_model_path: str = None,
        device: str = 'auto',
        sample_rate: int = 5
    ):
        """
        Args:
            yolo_model_path: YOLO 모델 파일 경로 (None이면 settings.YOLO_FACE_MODEL_PATH 사용)
            device: 'auto', 'cuda', 'cpu' 중 하나
            sample_rate: 몇 프레임마다 샘플링할지 (5 = 5프레임마다 1프레임 추출)
        """
        self.sample_rate = sample_rate

        # YOLO 모델 경로 설정 (None이면 settings에서 가져옴)
        if yolo_model_path is None:
            from django.conf import settings
            yolo_model_path = str(settings.YOLO_FACE_MODEL_PATH)

        # GPU 사용 가능 여부 확인
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # YOLO 모델 로드
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"YOLO model loaded from {yolo_model_path}")

        # FaceNet 모델 로드 (lazy loading - 필요할 때 로드)
        self.facenet_model = None
        self.mtcnn = None

    def _calculate_clarity(self, img: np.ndarray) -> float:
        """
        이미지 선명도 계산 (Laplacian Variance)
        - 정지해 있는(모션 블러가 없는) 이미지를 찾기 위함
        - 값이 클수록 선명함
        """
        try:
            if img.size == 0:
                return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0

    def _load_facenet(self):
        """FaceNet 모델 로드 (처음 사용 시 한 번만)"""
        if self.facenet_model is not None:
            return

        from facenet_pytorch import InceptionResnetV1, MTCNN

        # MTCNN: 얼굴 정렬 및 전처리
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            device=self.device,
            post_process=False  # 수동으로 정규화할 것임
        )

        # InceptionResnetV1: 얼굴 임베딩 추출 (512차원)
        self.facenet_model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)

        logger.info("FaceNet model loaded")

    def load_video_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, np.ndarray]]:
        """
        비디오에서 프레임 샘플링

        Args:
            video_path: 비디오 파일 경로
            max_frames: 최대 추출 프레임 수 (None = 제한 없음)

        Returns:
            List[(frame_idx, frame_array)]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Video: {total_frames} frames, {fps:.2f} FPS, "
            f"sampling every {self.sample_rate} frames"
        )

        sampled_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # sample_rate마다 프레임 추출
            if frame_idx % self.sample_rate == 0:
                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append((frame_idx, frame_rgb))

                if max_frames and len(sampled_frames) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(f"Sampled {len(sampled_frames)} frames")
        return sampled_frames

    def detect_faces_yolo(
        self,
        frames: List[Tuple[int, np.ndarray]],
        conf_threshold: float = 0.5
    ) -> List[DetectedFace]:
        """
        YOLO로 얼굴 감지

        Args:
            frames: [(frame_idx, frame_array), ...]
            conf_threshold: 신뢰도 임계값 (0.0 ~ 1.0)

        Returns:
            List[DetectedFace]
        """
        all_detected_faces = []

        for frame_idx, frame in frames:
            # YOLO 추론
            results = self.yolo_model(frame, verbose=False)

            # 결과 파싱
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue

                    # 바운딩 박스 좌표 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    # 유효성 검사
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 얼굴 영역 크롭
                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size == 0:
                        continue

                    # 선명도 계산
                    clarity = self._calculate_clarity(face_img)

                    detected_face = DetectedFace(
                        frame_idx=frame_idx,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        face_img=face_img,
                        clarity=clarity
                    )
                    all_detected_faces.append(detected_face)

        logger.info(f"Detected {len(all_detected_faces)} faces with YOLO")
        return all_detected_faces

    def extract_embeddings(
        self,
        detected_faces: List[DetectedFace]
    ) -> List[DetectedFace]:
        """
        FaceNet으로 얼굴 임베딩 추출

        Args:
            detected_faces: YOLO로 감지된 얼굴 리스트

        Returns:
            임베딩이 추가된 DetectedFace 리스트 (실패한 얼굴은 제외)
        """
        self._load_facenet()

        faces_with_embeddings = []

        for face in detected_faces:
            try:
                # YOLO 결과를 직접 사용 (MTCNN 건너뛰기)
                # YOLO가 감지한 얼굴 영역을 160x160으로 리사이즈
                import cv2
                face_resized = cv2.resize(face.face_img, (160, 160))

                # RGB 확인 (YOLO 출력은 이미 RGB)
                if face_resized.shape[2] != 3:
                    logger.debug(f"Invalid image shape at frame {face.frame_idx}")
                    continue

                # Tensor로 변환 및 정규화
                # (H, W, C) -> (C, H, W)
                face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()

                # 정규화 (FaceNet 입력 범위: -1 ~ 1)
                face_tensor = (face_tensor - 127.5) / 128.0

                # GPU로 이동
                face_tensor = face_tensor.unsqueeze(0).to(self.device)

                # 임베딩 추출 (512차원)
                with torch.no_grad():
                    embedding = self.facenet_model(face_tensor)
                    embedding = embedding.cpu().numpy().flatten()

                # L2 정규화 (코사인 유사도 계산을 위해)
                embedding = normalize(embedding.reshape(1, -1))[0]

                face.embedding = embedding
                faces_with_embeddings.append(face)

            except Exception as e:
                logger.warning(
                    f"Failed to extract embedding for face at "
                    f"frame {face.frame_idx}: {e}"
                )
                continue

        logger.info(
            f"Extracted embeddings for {len(faces_with_embeddings)} faces"
        )
        return faces_with_embeddings

    def cluster_faces(
        self,
        faces_with_embeddings: List[DetectedFace],
        eps: float = 0.5,
        min_samples: int = 2
    ) -> Dict[int, List[DetectedFace]]:
        """
        DBSCAN으로 얼굴 클러스터링 (동일 인물 그룹화)

        Args:
            faces_with_embeddings: 임베딩이 있는 얼굴 리스트
            eps: DBSCAN epsilon (거리 임계값, 작을수록 엄격)
            min_samples: 최소 샘플 수 (클러스터로 인정할 최소 얼굴 수)

        Returns:
            {cluster_id: [DetectedFace, ...], ...}
            cluster_id = -1은 노이즈 (어느 클러스터에도 속하지 않음)
        """
        if len(faces_with_embeddings) == 0:
            return {}

        # 임베딩 행렬 생성 (N x 512)
        embeddings = np.array([face.embedding for face in faces_with_embeddings])

        # DBSCAN 클러스터링 (코사인 거리 사용)
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        labels = clustering.fit_predict(embeddings)

        # 클러스터별로 얼굴 그룹화
        clusters = {}
        for face, label in zip(faces_with_embeddings, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(face)

        # 노이즈(-1)를 제외한 클러스터 수
        num_clusters = len([k for k in clusters.keys() if k != -1])
        num_noise = len(clusters.get(-1, []))

        logger.info(
            f"Clustering: {num_clusters} clusters, {num_noise} noise faces"
        )

        return clusters

    def select_best_thumbnail(
        self,
        cluster_faces: List[DetectedFace]
    ) -> DetectedFace:
        """
        클러스터 내에서 가장 좋은 썸네일 선택

        선택 기준 (가중치 적용):
        1. 선명도 (Clarity): 모션 블러가 없는 정지된 이미지 선호
        2. 크기 (Area): 너무 작은 얼굴보다는 큰 얼굴 선호
        3. 신뢰도 (Confidence): YOLO 감지 신뢰도

        점수 계산:
        Score = Clarity * sqrt(Area)
        (선명도를 최우선으로 하되, 크기가 너무 작으면 점수가 낮아짐)
        """
        def calculate_score(face: DetectedFace) -> float:
            # 면적의 제곱근 (선형적인 크기 비례)
            size_factor = np.sqrt(face.bbox_area)
            # 선명도 (노이즈로 인한 고주파 성분 제외를 위해 로그 스케일 고려 가능하나, 여기선 직접 곱함)
            # 아주 작은 얼굴의 노이즈가 선명도로 오인되는 것을 방지하기 위해 size_factor와 곱함
            return face.clarity * size_factor

        # 점수 기준 내림차순 정렬
        sorted_faces = sorted(
            cluster_faces,
            key=calculate_score,
            reverse=True
        )
        return sorted_faces[0]

    def save_thumbnail(
        self,
        face: DetectedFace,
        output_path: str,
        size: Tuple[int, int] = (160, 160)
    ) -> str:
        """
        썸네일 이미지 저장

        Args:
            face: DetectedFace 객체
            output_path: 저장 경로 (예: /app/media/faces/thumbnails/{video_id}/face_1.jpg)
            size: 리사이즈 크기 (width, height)

        Returns:
            저장된 파일 경로
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 리사이즈
        resized = cv2.resize(face.face_img, size, interpolation=cv2.INTER_AREA)

        # RGB → BGR 변환 (OpenCV 저장 형식)
        bgr_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        # 저장
        cv2.imwrite(output_path, bgr_img)
        logger.debug(f"Thumbnail saved: {output_path}")

        return output_path

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        eps: float = 0.5,
        min_samples: int = 2,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        전체 파이프라인 실행

        Args:
            video_path: 입력 비디오 경로
            output_dir: 썸네일 저장 디렉토리 (예: /app/media/faces/thumbnails/{video_id}/)
            eps: DBSCAN epsilon
            min_samples: DBSCAN 최소 샘플 수
            conf_threshold: YOLO 신뢰도 임계값

        Returns:
            [
                {
                    'face_index': 1,
                    'thumbnail_path': '/app/media/faces/...',
                    'embedding': [0.1, 0.2, ...],  # 512-dim list
                    'appearance_count': 5,
                    'first_frame': 0,
                    'last_frame': 100
                },
                ...
            ]
        """
        logger.info(f"Starting face detection pipeline for {video_path}")

        # 1. 프레임 샘플링
        frames = self.load_video_frames(video_path)

        if len(frames) == 0:
            logger.warning("No frames extracted")
            return []

        # 2. YOLO 얼굴 감지
        detected_faces = self.detect_faces_yolo(frames, conf_threshold)

        if len(detected_faces) == 0:
            logger.warning("No faces detected")
            return []

        # 3. FaceNet 임베딩 추출
        faces_with_embeddings = self.extract_embeddings(detected_faces)

        if len(faces_with_embeddings) == 0:
            logger.warning("No embeddings extracted")
            return []

        # 4. DBSCAN 클러스터링
        clusters = self.cluster_faces(
            faces_with_embeddings,
            eps=eps,
            min_samples=min_samples
        )

        # 5. 각 클러스터에서 대표 썸네일 선택 및 저장
        result_faces = []
        face_index = 1

        # 클러스터 ID 정렬 (노이즈 제외)
        cluster_ids = sorted([k for k in clusters.keys() if k != -1])

        for cluster_id in cluster_ids:
            cluster_faces = clusters[cluster_id]

            # 대표 얼굴 선택
            best_face = self.select_best_thumbnail(cluster_faces)

            # 썸네일 저장
            thumbnail_filename = f"face_{face_index}.jpg"
            thumbnail_path = os.path.join(output_dir, thumbnail_filename)
            self.save_thumbnail(best_face, thumbnail_path)

            # 프레임 정보 계산
            frame_indices = [f.frame_idx for f in cluster_faces]
            first_frame = min(frame_indices)
            last_frame = max(frame_indices)

            result_faces.append({
                'face_index': face_index,
                'thumbnail_path': thumbnail_path,
                'embedding': best_face.embedding.tolist(),  # numpy → list
                'appearance_count': len(cluster_faces),
                'first_frame': first_frame,
                'last_frame': last_frame
            })

            face_index += 1

        logger.info(f"Pipeline completed: {len(result_faces)} unique faces found")
        return result_faces
