# FaceBlur Project - 개발 진행 상황

**작성일**: 2025-11-19
**현재 Phase**: Phase 3 - 얼굴 감지 및 임베딩 파이프라인 (진행 중)

---

## 📋 전체 개발 로드맵

### ✅ Phase 1: Django 백엔드 기본 구축 (완료)
- [x] 개발 환경 설정 (requirements.txt, .env, .gitignore)
- [x] Docker 환경 구성 (docker-compose.yml, Dockerfile.django)
- [x] Django 앱 3개 생성 (accounts, videos, processing)
- [x] 데이터베이스 모델 구현 (Video, Face, ProcessingJob)
- [x] Django settings.py 완전 업데이트
- [x] DRF Serializers 작성 (Video, Face, ProcessingJob)
- [x] API Views 구현 (VideoViewSet, FaceViewSet)
- [x] URL 라우팅 설정
- [x] HTML 템플릿 통합 (upload, select_faces, preview)
- [x] 마이그레이션 실행 및 테스트

### ✅ Phase 2: 파일 업로드 기능 및 프론트엔드 (완료)
- [x] 파일 업로드 API 구현
- [x] 업로드 페이지 UI 구현 (`templates/videos/upload.html`)
- [x] 프론트엔드 JS 로직 구현 (`static/js/upload.js`)
- [x] 영상 메타데이터 추출 (MoviePy 연동)
- [x] 진행률 표시 연동
- [ ] S3 연동 (현재 로컬 저장)

### ⏳ Phase 3: AI 얼굴 감지 및 임베딩 파이프라인 (진행 중 - 90%)
- [x] YOLO (YOLOv11) 얼굴 감지 및 추적
- [x] InsightFace (ArcFace) 임베딩 추출
- [x] AgglomerativeClustering (HAC) 동일 인물 그룹화
- [x] Celery 비동기 작업 연동
- [x] 썸네일 생성 및 저장
- [ ] 임베딩 추출 성공률 최종 검증

### ⏳ Phase 4: 영상 블러 처리
- [ ] 선택된 얼굴 블러 처리 구현
- [ ] 처리된 영상 저장
- [ ] 다운로드 기능

### ⏳ Phase 5: Docker Compose 통합 & 배포
- [ ] 전체 서비스 통합 테스트
- [ ] AWS EC2 배포
- [ ] S3 연동
- [ ] 운영 환경 설정

---

## 🔧 해결된 문제들

### 1. Docker 빌드 이슈
**문제**: `ModuleNotFoundError: No module named 'django'` / `'insightface'`
**원인**: Docker 이미지가 새 의존성으로 빌드되지 않음
**해결**:
```bash
docker-compose build --no-cache django
docker-compose build --no-cache celery_worker
```

### 2. process_video 파라미터 불일치
**문제**: `FaceDetectionPipeline.process_video() got an unexpected keyword argument 'eps'`
**원인**: face_detection.py API 변경 후 tasks.py 미수정
**해결**: tasks.py 라인 160-166 수정
```python
# Before (DBSCAN 방식)
detected_faces = pipeline.process_video(
    video_path=video_path,
    output_dir=thumbnail_dir,
    eps=0.5,
    min_samples=2,
    conf_threshold=0.5
)

# After (HAC 방식)
detected_faces = pipeline.process_video(
    video_path=video_path,
    output_dir=thumbnail_dir,
    conf_threshold=0.5,
    sim_threshold=0.6
)
```

### 3. 단일 Tracklet 클러스터링 에러
**문제**: `ValueError: Found array with 1 sample(s) while a minimum of 2 is required`
**원인**: AgglomerativeClustering은 최소 2개 샘플 필요
**해결**: face_detection.py 라인 274-287에 조건 추가
```python
if len(valid_tracklets) == 1:
    logger.info("Only 1 tracklet found, skipping clustering.")
    labels = [0]
else:
    clustering = AgglomerativeClustering(...)
    labels = clustering.fit_predict(embeddings)
```

### 4. 임베딩 추출 실패 (99.9% 실패율)
**문제**: `Embedding extraction: 1 success, 839 fail`
**원인**:
- InsightFace det_size=(640,640)이 작은 crop 이미지에 부적합
- crop 이미지 최소 크기가 너무 작음

**해결**:
1. det_size 축소: (640, 640) → (160, 160)
   - face_detection.py 라인 120
2. crop 최소 크기 증가: 160 → 256
   - face_detection.py 라인 222

---

## 🔄 알고리즘 파이프라인

### 전체 흐름
```
1. YOLO Tracking (프레임별 얼굴 감지)
   ↓
2. 각 얼굴 crop 이미지 생성
   ↓
3. 선명도(clarity) 계산 ← Laplacian Variance 적용
   ↓
4. InsightFace 임베딩 추출
   ↓
5. Tracklet별 평균 임베딩 계산
   ↓
6. HAC 클러스터링 (동일 인물 그룹화)
   ↓
7. 클러스터별 최적 썸네일 선택 ← clarity 값 사용
   ↓
8. 썸네일 저장
```

### 선명도 계산 (Laplacian Variance)
- **위치**: face_detection.py 라인 126-134
- **적용 시점**: 임베딩 추출 전 (crop 직후)
- **사용 시점**: 클러스터링 완료 후 썸네일 선택 시
- **공식**: `score = clarity * sqrt(bbox_area)`

---

## 📂 주요 파일 변경사항

### apps/videos/face_detection.py
- InsightFace det_size: 640 → 160 (라인 120)
- crop min_size: 160 → 256 (라인 222)
- 임베딩 성공/실패 카운터 추가 (라인 162-164)
- 단일 tracklet 처리 로직 추가 (라인 274-287)

### apps/videos/tasks.py
- process_video 파라미터 수정 (라인 160-166)
- eps, min_samples → conf_threshold, sim_threshold

### Dockerfile.django
- g++, python3-dev 추가
- numpy, cython 사전 설치

### requirements.txt
- insightface>=0.7.3
- onnxruntime>=1.16.0

---

## 📂 현재 프로젝트 구조

```
face_blur_project/
├── apps/                           # Django 앱들
│   ├── accounts/                   # ✅ 사용자 관리 앱
│   ├── videos/                     # ✅ 영상 관리 앱
│   │   ├── models.py              # Video, Face, ProcessingJob 모델
│   │   ├── serializers.py         # DRF Serializers
│   │   ├── views.py               # API & Template Views
│   │   ├── urls.py                # URL 라우팅
│   │   ├── face_detection.py      # ✅ 얼굴 감지 파이프라인 (YOLO + InsightFace)
│   │   └── tasks.py               # ✅ Celery 비동기 작업
│   │
│   └── processing/                 # 영상 처리 앱
│
├── face_blur_web/                  # Django 프로젝트 설정
│   ├── settings.py                # 설정
│   ├── celery.py                  # ✅ Celery 설정
│   └── urls.py                    # 메인 라우팅
│
├── templates/                      # HTML 템플릿
│   ├── base.html                  # 기본 레이아웃
│   └── videos/
│       ├── upload.html            # 업로드 페이지
│       ├── select_faces.html      # 얼굴 선택 페이지
│       └── preview.html           # 미리보기 페이지
│
├── static/                         # 정적 파일
│   └── js/
│       ├── api.js                 # API 통신 유틸리티
│       ├── upload.js              # 업로드 로직
│       ├── select-faces.js        # 얼굴 선택 로직
│       └── preview.js             # 미리보기 로직
│
├── media/                          # 업로드 파일 저장
├── models/                         # AI 모델 파일
│   └── yolo11n-face.pt           # YOLO 얼굴 감지 모델
├── docker-compose.yml              # Docker 서비스 정의
├── Dockerfile.django               # Django 컨테이너
└── manage.py                       # Django 관리 스크립트
```

---

## 🎯 현재 상태 요약

| 항목 | 상태 | 완성도 |
|------|------|--------|
| **Phase 1 (Backend Basic)** | ✅ | **100%** |
| **Phase 2 (Upload & FE)** | ✅ | **95%** |
| **Phase 3 (AI Pipeline)** | ⏳ | **90%** |
| **Phase 4 (Blur Processing)** | ❌ | 0% |
| **Phase 5 (Deploy)** | ❌ | 0% |

---

## 🚀 다음 단계

1. 새 비디오 업로드하여 임베딩 추출 성공률 확인
2. 얼굴 감지 및 클러스터링 결과 검증
3. Phase 4: 실제 블러 처리 구현

---

## 🔧 명령어 참고

### Docker 관리
```bash
# 전체 재빌드
docker-compose build --no-cache

# 특정 서비스 재빌드
docker-compose build --no-cache celery_worker

# 서비스 재시작
docker-compose restart celery_worker

# 로그 확인
docker-compose logs -f celery_worker
```

### 테스트
```bash
# 비디오 업로드 후 로그에서 확인할 내용
INFO: Embedding extraction: XXX success, YY fail
INFO: Pipeline completed: N unique faces found
```

---

**마지막 업데이트**: 2025-11-19
**작성자**: Antigravity
