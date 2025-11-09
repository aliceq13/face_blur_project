# FaceBlur - AI 기반 영상 얼굴 블러 처리 시스템

**영상 속 특정 인물의 얼굴을 선택적으로 블러 처리하는 웹 서비스**

[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://www.djangoproject.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-blue.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-orange.svg)](https://aws.amazon.com/)

---

## 📖 프로젝트 소개

FaceBlur는 **AI 얼굴 인식 기술**을 활용하여 영상 속 얼굴을 자동으로 감지하고, 사용자가 선택한 얼굴만 블러 처리하는 웹 서비스입니다.

### 🎯 주요 기능

1. **영상 업로드**: MP4, MOV, AVI 형식 지원 (최대 2GB)
2. **자동 얼굴 분석**: YOLOv8 + FaceNet 기반 얼굴 검출 및 클러스터링
3. **선택적 블러 처리**: 사용자가 원하는 얼굴만 선택하여 블러 적용
4. **실시간 진행률**: WebSocket을 통한 처리 상태 실시간 업데이트
5. **클라우드 저장**: AWS S3를 통한 안전한 파일 관리

---

## 🏗️ 시스템 아키텍처

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   사용자     │────▶│    Django    │────▶│   FastAPI    │
│  (브라우저)  │     │  웹 서버      │     │  AI 서버     │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                             │                     │
                      ┌──────┴────────┐    ┌──────┴───────┐
                      │  PostgreSQL   │    │   YOLOv8     │
                      │  (데이터베이스)│    │   FaceNet    │
                      └───────────────┘    └──────────────┘
                             │
                      ┌──────┴────────┐
                      │     Redis     │
                      │  (캐시/Celery)│
                      └───────────────┘
```

### 기술 스택

**Frontend**:
- HTML5, CSS3, JavaScript
- Tailwind CSS

**Backend**:
- Django 4.2 (웹 프레임워크)
- Django REST Framework (API)
- FastAPI (AI 모델 서빙)
- Celery (비동기 작업 처리)

**AI/ML**:
- YOLOv8 (얼굴 검출)
- FaceNet (얼굴 임베딩)
- OpenCV (영상 처리)
- DBSCAN (얼굴 클러스터링)

**Database**:
- PostgreSQL 15 (운영)
- SQLite (개발)
- Redis 7 (캐시 & Celery 브로커)

**Infrastructure**:
- Docker & Docker Compose
- AWS S3 (파일 저장)
- AWS EC2 (서버 호스팅)
- Nginx (리버스 프록시)

---

## 🚀 빠른 시작

### 전제 조건

- Docker Desktop 설치
- Git 설치
- (선택) Python 3.11+ 설치

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/yourusername/face_blur_project.git
cd face_blur_project
```

### 2️⃣ 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (선택사항)
# 기본 설정으로도 로컬 개발 가능
```

### 3️⃣ Docker Compose 실행

```bash
# 서비스 빌드 및 시작
docker-compose up -d --build

# 로그 확인
docker-compose logs -f django
```

### 4️⃣ 마이그레이션 실행

```bash
# 데이터베이스 마이그레이션
docker-compose exec django python manage.py migrate

# 슈퍼유저 생성
docker-compose exec django python manage.py createsuperuser
```

### 5️⃣ 접속

- **웹 서비스**: http://localhost:8000
- **Admin 페이지**: http://localhost:8000/admin
- **API 문서**: http://localhost:8000/api/

---

## 📁 프로젝트 구조

```
face_blur_project/
├── apps/                       # Django 앱들
│   ├── accounts/               # 사용자 관리
│   ├── videos/                 # 영상 및 얼굴 관리
│   └── processing/             # 영상 처리 로직
│
├── face_blur_web/              # Django 프로젝트 설정
│   ├── settings.py             # 프로젝트 설정
│   └── urls.py                 # URL 라우팅
│
├── templates/                  # HTML 템플릿
├── static/                     # 정적 파일 (CSS, JS)
├── media/                      # 업로드된 파일
│
├── docker-compose.yml          # Docker 서비스 정의
├── Dockerfile.django           # Django 컨테이너
├── Dockerfile                  # FastAPI AI 서버 (향후)
│
├── requirements.txt            # Python 패키지
├── .env.example                # 환경 변수 템플릿
│
├── SKILL.md                    # 전체 아키텍처 가이드
├── PROGRESS.md                 # 개발 진행 상황
├── QUICKSTART.md               # 빠른 시작 가이드
└── README.md                   # 이 파일
```

---

## 🔧 개발 가이드

### 로컬 개발 환경 설정

```bash
# Docker 서비스 시작
docker-compose up -d

# Django 쉘 접속
docker-compose exec django python manage.py shell

# 테스트 실행
docker-compose exec django python manage.py test

# 코드 스타일 검사 (향후 추가)
# docker-compose exec django flake8
```

### 데이터베이스 마이그레이션

```bash
# 마이그레이션 파일 생성
docker-compose exec django python manage.py makemigrations

# 마이그레이션 적용
docker-compose exec django python manage.py migrate

# 마이그레이션 상태 확인
docker-compose exec django python manage.py showmigrations
```

### 정적 파일 관리

```bash
# Static 파일 수집
docker-compose exec django python manage.py collectstatic --noinput

# 개발 서버에서는 자동으로 제공됨
```

---

## 📊 데이터 모델

### Video (영상)
- 업로드된 영상 정보
- 상태: uploaded → analyzing → ready → processing → completed
- S3 URL, 메타데이터 (해상도, FPS, 길이)

### Face (얼굴)
- 영상에서 발견된 고유 얼굴
- 썸네일, 512차원 임베딩 벡터
- 블러 처리 여부 (사용자 선택)

### ProcessingJob (처리 작업)
- Celery 비동기 작업 추적
- 얼굴 분석, 영상 처리 상태

자세한 내용은 `SKILL.md` 참조

---

## 🎨 사용자 워크플로우

1. **영상 업로드**
   - 사용자가 영상 파일 업로드
   - 서버가 S3에 저장 및 메타데이터 추출

2. **얼굴 분석** (자동)
   - YOLOv8로 프레임별 얼굴 검출
   - FaceNet으로 임베딩 추출
   - DBSCAN 클러스터링으로 고유 얼굴 그룹화

3. **얼굴 선택**
   - 사용자가 블러 처리할 얼굴 선택
   - 썸네일 그리드에서 체크박스로 선택

4. **영상 처리** (비동기)
   - Celery 작업으로 전체 프레임 처리
   - 선택된 얼굴만 OpenCV로 블러 적용
   - 처리된 영상 S3에 저장

5. **다운로드**
   - 처리 완료된 영상 미리보기
   - 다운로드 또는 공유

---

## 🔐 보안

### 인증
- Django 세션 기반 인증
- (향후) JWT 토큰 인증 추가 예정

### 파일 보안
- S3 버킷: Private 권한
- Presigned URL로 임시 접근
- 7일 후 자동 삭제

### API 보안
- CORS 설정
- CSRF 보호
- Rate Limiting (향후)

---

## 🌐 배포 (AWS)

### 필요한 AWS 리소스

1. **EC2 인스턴스**
   - Django 웹 서버: t3.large
   - FastAPI AI 서버: g4dn.xlarge (GPU)
   - Celery Worker: t3.xlarge

2. **RDS (PostgreSQL)**
   - db.t3.medium
   - 스토리지: 100GB

3. **ElastiCache (Redis)**
   - cache.t3.medium

4. **S3 버킷**
   - 영상 파일 저장

5. **기타**
   - Application Load Balancer
   - Route 53 (DNS)
   - Certificate Manager (SSL)

자세한 배포 가이드는 `SKILL.md` 참조

---

## 📈 개발 진행 상황

**현재 Phase**: Phase 1 - Django 백엔드 기본 구축 (80% 완료)

✅ **완료**:
- Docker 환경 구성
- Django 앱 생성 (accounts, videos, processing)
- 데이터베이스 모델 구현
- Admin 페이지 구현
- 설정 파일 완성

⏳ **진행 중**:
- DRF Serializers
- API Views
- URL 라우팅
- 템플릿 통합

📋 **다음 단계**:
- Phase 2: 파일 업로드 기능
- Phase 3: FastAPI AI 서버
- Phase 4: Celery 비동기 작업
- Phase 5: AWS 배포

자세한 진행 상황은 `PROGRESS.md` 참조

---

## 🤝 기여하기

현재 개인 프로젝트로 진행 중입니다.

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 📧 문의

- **Email**: your-email@example.com
- **GitHub**: https://github.com/yourusername

---

## 📚 문서

- **SKILL.md**: 전체 아키텍처 및 기술 상세 가이드 (1,890줄)
- **PROGRESS.md**: 개발 진행 상황 및 완료 작업
- **QUICKSTART.md**: 빠른 시작 및 명령어 치트시트
- **.env.example**: 환경 변수 설정 가이드

---

## 🙏 감사의 말

이 프로젝트는 다음 기술들을 활용합니다:
- [Django](https://www.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [FaceNet](https://github.com/davidsandberg/facenet)
- [OpenCV](https://opencv.org/)

---

**마지막 업데이트**: 2025-11-09
**버전**: 0.1.0 (Phase 1)
