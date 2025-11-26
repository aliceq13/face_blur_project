#!/usr/bin/env python
"""
YOLO 얼굴 검출 테스트 스크립트
- Bounding box를 그려서 YOLO 검출 성능 확인
- Track ID 표시
- 검출 통계 출력
"""

import os
import sys
import cv2
import torch
from pathlib import Path
from collections import defaultdict

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
import django
django.setup()

from django.conf import settings
from ultralytics import YOLO


def test_face_detection(video_path: str, output_path: str = None):
    """
    YOLO로 얼굴 검출하고 bounding box 그린 영상 생성

    Args:
        video_path: 입력 영상 경로
        output_path: 출력 영상 경로 (기본: media/test_output/detection_result.mp4)
    """
    # 출력 경로 설정
    if output_path is None:
        output_dir = Path(settings.MEDIA_ROOT) / 'test_output'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'detection_result.mp4'

    print("=" * 80)
    print("YOLO Face Detection Test")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print()

    # YOLO 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    yolo_model_path = settings.YOLO_FACE_MODEL_PATH
    print(f"Loading YOLO model: {yolo_model_path}")
    model = YOLO(str(yolo_model_path))
    model.to(device)
    print("✓ YOLO model loaded")
    print()

    # 영상 정보 읽기
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()

    cap.release()

    # 출력 영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # YOLO 추적 실행
    print("Running YOLO tracking...")
    results = model.track(
        source=str(video_path),
        conf=0.5,
        persist=True,
        verbose=False,
        stream=True,
        device=device,
        half=True,
        imgsz=640,
        batch=8
    )

    # 통계 변수
    frame_idx = 0
    total_detections = 0
    track_stats = defaultdict(int)  # {track_id: 검출 횟수}
    frames_with_faces = 0

    # 프레임별 처리
    for result in results:
        frame = result.orig_img.copy()  # BGR 이미지

        # 검출 결과가 있으면 bounding box 그리기
        if result.boxes:
            frames_with_faces += 1
            num_faces = len(result.boxes)
            total_detections += num_faces

            for box in result.boxes:
                # Bounding box 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf.item())

                # Track ID (없으면 -1)
                track_id = int(box.id.item()) if box.id is not None else -1
                if track_id != -1:
                    track_stats[track_id] += 1

                # Bounding box 그리기 (초록색)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Track ID와 confidence 표시
                label = f"ID:{track_id} ({conf:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 라벨 배경 (검은색)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 0, 0), -1)

                # 라벨 텍스트 (흰색)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 프레임 정보 표시 (좌측 상단)
        info_text = f"Frame: {frame_idx}/{total_frames} | Faces: {len(result.boxes) if result.boxes else 0}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 영상에 프레임 쓰기
        out.write(frame)

        frame_idx += 1

        # 진행률 표시 (10% 단위)
        if frame_idx % max(1, total_frames // 10) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

    out.release()

    # 통계 출력
    print()
    print("=" * 80)
    print("Detection Statistics")
    print("=" * 80)
    print(f"Total frames processed: {frame_idx}")
    print(f"Frames with faces: {frames_with_faces} ({frames_with_faces/frame_idx*100:.1f}%)")
    print(f"Total face detections: {total_detections}")
    print(f"Average faces per frame: {total_detections/frame_idx:.2f}")
    print(f"Unique track IDs: {len(track_stats)}")
    print()

    # Track ID별 통계 (상위 10개)
    print("Top 10 Track IDs by detection count:")
    sorted_tracks = sorted(track_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for track_id, count in sorted_tracks:
        print(f"  Track ID {track_id}: {count} frames ({count/frame_idx*100:.1f}%)")

    print()
    print("=" * 80)
    print(f"✓ Output saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_face_detection.py <video_path> [output_path]")
        print()
        print("Example:")
        print("  python test_face_detection.py media/videos/test.mp4")
        print("  python test_face_detection.py media/videos/test.mp4 media/test_output/result.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(video_path):
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)

    test_face_detection(video_path, output_path)
