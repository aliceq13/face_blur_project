# -*- coding: utf-8 -*-
"""
비디오 처리 Celery Tasks

이 모듈은 오래 걸리는 비디오 처리 작업을 백그라운드에서 비동기적으로 실행합니다.

Tasks:
- analyze_faces_task: 얼굴 분석 (YOLO + FaceNet + DBSCAN)
- process_video_blur_task: 비디오 블러 처리 (Phase 4에서 구현)
"""

import os
import logging
from celery import shared_task
from django.conf import settings
from .models import Video, Face, ProcessingJob
from .face_detection import FaceDetectionPipeline

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def analyze_faces_task(self, video_id: str):
    """
    비디오 얼굴 분석 Celery Task

    처리 흐름:
    1. Video 객체 로드
    2. 상태를 'analyzing'으로 변경
    3. ProcessingJob 생성/업데이트
    4. FaceDetectionPipeline 실행
       - 프레임 샘플링 (5프레임마다)
       - YOLO 얼굴 감지
       - FaceNet 임베딩 추출
       - DBSCAN 클러스터링
       - 썸네일 저장
    5. Face 모델 생성 (각 클러스터마다)
    6. Video 상태를 'ready'로 변경
    7. ProcessingJob 완료 처리

    Args:
        video_id: Video 모델 UUID (문자열)

    Returns:
        {
            'video_id': str,
            'faces_count': int,
            'status': 'success'
        }

    Raises:
        ValueError: Video를 찾을 수 없는 경우
        Exception: 처리 중 오류 발생 시 재시도
    """
    logger.info(f"[Task {self.request.id}] Starting face analysis for video {video_id}")

    try:
        # ====================================================================
        # 1. Video 객체 로드
        # ====================================================================
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            logger.error(f"Video {video_id} not found")
            raise ValueError(f"Video {video_id} does not exist")

        # ====================================================================
        # 2. 상태 업데이트: uploaded → analyzing
        # ====================================================================
        video.status = 'analyzing'
        video.progress = 0
        video.save(update_fields=['status', 'progress'])

        logger.info(f"Video {video_id} status changed to 'analyzing'")

        # ====================================================================
        # 3. ProcessingJob 생성 또는 업데이트
        # ====================================================================
        job, created = ProcessingJob.objects.get_or_create(
            video=video,
            job_type='face_analysis',
            celery_task_id=self.request.id,
            defaults={
                'status': 'started',
                'progress': 0
            }
        )

        if not created:
            job.status = 'started'
            job.progress = 0
            job.celery_task_id = self.request.id
            job.save(update_fields=['status', 'progress', 'celery_task_id'])

        logger.info(f"ProcessingJob created/updated: {job.id}")

        # ====================================================================
        # 4. 비디오 파일 경로 확인
        # ====================================================================
        # Video 모델의 original_file_url은 URLField이므로
        # 로컬 파일 시스템 경로로 변환 필요
        # URL 형식: /media/videos/{uuid}.mp4
        # 실제 경로: /app/media/videos/{uuid}.mp4

        video_filename = video.original_file_url.split('/')[-1]
        video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Video file path: {video_path}")

        # ====================================================================
        # 5. 출력 디렉토리 설정
        # ====================================================================
        # 썸네일 저장 디렉토리: /app/media/faces/thumbnails/{video_id}/
        thumbnail_dir = os.path.join(
            settings.MEDIA_ROOT,
            'faces',
            'thumbnails',
            str(video.id)
        )
        os.makedirs(thumbnail_dir, exist_ok=True)

        # ====================================================================
        # 5-1. YOLO 모델 파일 검증
        # ====================================================================
        yolo_model_path = str(settings.YOLO_FACE_MODEL_PATH)
        if not os.path.exists(yolo_model_path):
            error_msg = f"YOLO model file not found: {yolo_model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        model_size = os.path.getsize(yolo_model_path)
        if model_size < 1_000_000:  # Less than 1MB
            error_msg = f"YOLO model file seems corrupted (size: {model_size} bytes)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"YOLO model validated: {yolo_model_path} ({model_size / 1024 / 1024:.1f} MB)")

        # ====================================================================
        # 6. FaceDetectionPipeline 실행
        # ====================================================================
        logger.info("Initializing FaceDetectionPipeline...")
        pipeline = FaceDetectionPipeline(
            # yolo_model_path는 None으로 설정하여 settings.YOLO_FACE_MODEL_PATH 사용
            yolo_model_path=None,
            device='auto',  # CUDA 사용 가능하면 자동으로 사용
            sample_rate=1   # 모든 프레임 분석 (1프레임마다)
        )

        # 진행률 업데이트: 10%
        video.progress = 10
        video.save(update_fields=['progress'])
        job.progress = 10
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 10})

        logger.info("Running face detection pipeline...")
        detected_faces = pipeline.process_video(
            video_path=video_path,
            output_dir=thumbnail_dir,
            conf_threshold=0.5,  # YOLO 신뢰도 임계값
            sim_threshold=0.6    # ArcFace 유사도 임계값 (같은 사람 판단)
        )

        logger.info(f"Pipeline completed: {len(detected_faces)} unique faces found")

        # 진행률 업데이트: 70%
        video.progress = 70
        video.save(update_fields=['progress'])
        job.progress = 70
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 70})

        # ====================================================================
        # 7. Face 모델 생성
        # ====================================================================
        logger.info("Creating Face model instances...")
        created_faces = []

        for face_data in detected_faces:
            # thumbnail_path를 URLField 형식으로 변환
            # /app/media/faces/thumbnails/{video_id}/face_1.jpg
            # → /media/faces/thumbnails/{video_id}/face_1.jpg
            thumbnail_rel_path = face_data['thumbnail_path'].replace(
                str(settings.MEDIA_ROOT),
                settings.MEDIA_URL.rstrip('/')
            )

            face = Face.objects.create(
                video=video,
                face_index=face_data['face_index'],
                thumbnail_url=thumbnail_rel_path,
                embedding=face_data['embedding'],  # JSON 필드 (list)
                appearance_count=face_data['appearance_count'],
                first_frame=face_data['first_frame'],
                last_frame=face_data['last_frame'],
                is_blurred=True  # 기본값: 블러 처리 대상
            )
            created_faces.append(face)

        logger.info(f"Created {len(created_faces)} Face instances")

        # 진행률 업데이트: 90%
        video.progress = 90
        video.save(update_fields=['progress'])
        job.progress = 90
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 90})

        # ====================================================================
        # 8. Video 상태 업데이트: analyzing → ready
        # ====================================================================
        video.status = 'ready'
        video.progress = 100
        video.save(update_fields=['status', 'progress'])

        logger.info(f"Video {video_id} status changed to 'ready'")

        # ====================================================================
        # 9. ProcessingJob 완료 처리
        # ====================================================================
        job.status = 'success'
        job.progress = 100
        job.result_data = {
            'faces_count': len(created_faces),
            'face_ids': [str(f.id) for f in created_faces]
        }
        job.save(update_fields=['status', 'progress', 'result_data'])

        logger.info(
            f"[Task {self.request.id}] Face analysis completed successfully "
            f"for video {video_id}"
        )

        return {
            'video_id': str(video.id),
            'faces_count': len(created_faces),
            'status': 'success'
        }

    except (FileNotFoundError, ValueError) as exc:
        # ====================================================================
        # 복구 불가능한 에러 (파일 없음, 모델 손상 등)
        # ====================================================================
        logger.error(
            f"[Task {self.request.id}] Unrecoverable error for video {video_id}: {exc}",
            exc_info=True
        )

        # Video 상태 업데이트
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.save(update_fields=['status'])
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

        # ProcessingJob 업데이트
        try:
            job = ProcessingJob.objects.filter(
                video_id=video_id,
                job_type='face_analysis'
            ).order_by('-started_at').first()

            if job:
                job.status = 'failure'
                job.error_message = str(exc)
                job.save(update_fields=['status', 'error_message'])
        except Exception as e:
            logger.error(f"Failed to update ProcessingJob: {e}")

        # 재시도하지 않고 즉시 실패 처리
        raise exc

    except Exception as exc:
        # ====================================================================
        # 일시적 에러 (네트워크, GPU 메모리 등) - 재시도 가능
        # ====================================================================
        logger.error(
            f"[Task {self.request.id}] Face analysis failed for video {video_id}: {exc}",
            exc_info=True
        )

        # Video 상태 업데이트
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.save(update_fields=['status'])
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

        # ProcessingJob 업데이트
        try:
            job = ProcessingJob.objects.filter(
                video_id=video_id,
                job_type='face_analysis'
            ).order_by('-started_at').first()

            if job:
                job.status = 'failure'
                job.error_message = str(exc)
                job.save(update_fields=['status', 'error_message'])
        except Exception as e:
            logger.error(f"Failed to update ProcessingJob: {e}")

        # Celery 재시도 (최대 3회, 지수 백오프)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@shared_task(bind=True, max_retries=3)
def process_video_blur_task(self, video_id: str):
    """
    비디오 블러 처리 Celery Task (Phase 4에서 구현 예정)

    처리 흐름:
    1. Video 및 Face 데이터 로드
    2. is_blurred=True인 얼굴들의 위치 정보 수집
    3. 비디오 전체 프레임 순회하며 블러 처리
    4. 처리된 비디오 저장
    5. Video.processed_file_url 업데이트
    6. 상태를 'completed'로 변경

    Args:
        video_id: Video 모델 UUID (문자열)

    Returns:
        {
            'video_id': str,
            'processed_file_url': str,
            'status': 'success'
        }
    """
    logger.info(f"[Task {self.request.id}] Video blur processing not implemented yet")

    # TODO: Phase 4에서 구현
    # - OpenCV 또는 FFmpeg으로 프레임별 블러 처리
    # - 진행률 업데이트
    # - ProcessingJob 추적

    raise NotImplementedError("Video blur processing will be implemented in Phase 4")
