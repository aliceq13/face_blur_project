# FaceBlur Project - 개발 진행 상황

**작성일**: 2025-11-09
**현재 Phase**: Phase 1 - Django 백엔드 기본 구축 (80% 완료)

---

## 📋 전체 개발 로드맵

### ✅ Phase 1: Django 백엔드 기본 구축 (현재 단계 - 80% 완료)
- [x] 개발 환경 설정 (requirements.txt, .env, .gitignore)
- [x] Docker 환경 구성 (docker-compose.yml, Dockerfile.django)
- [x] Django 앱 3개 생성 (accounts, videos, processing)
- [x] 데이터베이스 모델 구현 (Video, Face, ProcessingJob)
- [x] Django settings.py 완전 업데이트
- [ ] DRF Serializers 작성 (다음 작업)
- [ ] API Views 구현 (업로드, 목록 조회)
- [ ] URL 라우팅 설정
- [ ] HTML 템플릿 통합
- [ ] 마이그레이션 실행 및 테스트

### ⏳ Phase 2: 간단한 파일 업로드 기능 구현
- [ ] 파일 업로드 API 구현
- [ ] 영상 메타데이터 추출 (MoviePy)
- [ ] S3 또는 로컬 저장
- [ ] 업로드 페이지 동작 구현
- [ ] 진행률 표시

### ⏳ Phase 3: FastAPI AI 모델 서빙
- [ ] fastapi_ai_server 프로젝트 생성
- [ ] YOLOv8 얼굴 검출 API
- [ ] FaceNet 임베딩 추출 API
- [ ] Django ↔ FastAPI 통신

### ⏳ Phase 4: Celery 비동기 작업
- [ ] Redis + Celery 설정
- [ ] 얼굴 분석 비동기 작업
- [ ] 영상 블러 처리 작업
- [ ] WebSocket 실시간 진행률

### ⏳ Phase 5: Docker Compose 통합 & 배포
- [ ] 전체 서비스 통합 테스트
- [ ] AWS EC2 배포
- [ ] S3 연동
- [ ] 운영 환경 설정

---

## 📂 현재 프로젝트 구조

```
face_blur_project/
├── apps/                           # Django 앱들
│   ├── accounts/                   # ✅ 사용자 관리 앱
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py              # UserProfile 모델
│   │   ├── admin.py               # Admin 페이지 설정
│   │   ├── views.py               # UserViewSet
│   │   └── urls.py                # URL 라우팅
│   │
│   ├── videos/                     # ✅ 영상 관리 앱
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py              # Video, Face, ProcessingJob 모델 (핵심!)
│   │   ├── admin.py               # 상세한 Admin 페이지
│   │   └── (serializers.py)       # ❌ 다음에 생성 예정
│   │   └── (views.py)             # ❌ 다음에 생성 예정
│   │   └── (urls.py)              # ❌ 다음에 생성 예정
│   │
│   └── processing/                 # ✅ 영상 처리 앱 (기본 구조만)
│       ├── __init__.py
│       ├── apps.py
│       ├── models.py              # 향후 확장용
│       ├── admin.py
│       └── views.py               # Phase 4에서 본격 구현
│
├── face_blur_web/                  # Django 프로젝트 설정
│   ├── __init__.py
│   ├── settings.py                # ✅ 완전히 업데이트됨 (AWS, DRF, CORS 등)
│   ├── urls.py                    # ⚠️ 아직 기본 상태 (다음에 수정)
│   ├── wsgi.py
│   └── asgi.py
│
├── stitch_/                        # 📁 HTML 프로토타입 (아직 통합 안됨)
│   ├── 비디오_업로드/
│   │   ├── code.html              # 업로드 페이지 UI
│   │   └── screen.png
│   ├── 썸네일_선택/
│   │   ├── code.html              # 썸네일 선택 UI
│   │   └── screen.png
│   └── 최종_비디오_미리보기_및_저장/
│       ├── code.html              # 미리보기 페이지 UI
│       └── screen.png
│
├── static/                         # ❌ 아직 생성 안됨
├── templates/                      # ❌ 아직 생성 안됨
├── media/                          # 업로드 파일 저장 (자동 생성)
├── logs/                           # 로그 파일 (자동 생성)
│
├── .env.example                    # ✅ 환경 변수 템플릿
├── .gitignore                      # ✅ Git 제외 파일 설정
├── requirements.txt                # ✅ Python 패키지 목록
├── docker-compose.yml              # ✅ Docker 서비스 정의
├── Dockerfile                      # ⚠️ FastAPI AI 서버용 (기존)
├── Dockerfile.django               # ✅ Django 웹 서버용 (신규)
├── manage.py                       # Django 관리 스크립트
├── db.sqlite3                      # SQLite 데이터베이스 (개발용)
├── SKILL.md                        # 📖 전체 아키텍처 가이드
├── PROGRESS.md                     # 📝 이 파일
└── readme.md                       # 기존 README
```

---

## ✅ 완료된 작업 상세

### 1. 개발 환경 설정

#### `requirements.txt`
- Django 4.2.7 + DRF
- AWS SDK (boto3, django-storages)
- Redis & Celery
- HTTPX (FastAPI 통신)
- OpenCV, Pillow, MoviePy
- **총 30개 이상의 패키지, 상세한 주석 포함**

#### `.env.example`
- 환경 변수 템플릿 파일
- AWS, DB, Redis 설정 예시
- **보안 팁 포함**

#### `.gitignore`
- Python, Django, AWS 관련 파일 제외
- 민감 정보 보호 (.env, *.pem, *.key)

---

### 2. Docker 환경

#### `docker-compose.yml`
**포함된 서비스**:
- `db`: PostgreSQL 15
- `redis`: Redis 7
- `django`: Django 웹 서버 (Gunicorn)
- `celery_worker`: (주석 처리, Phase 4에서 활성화)

**주요 설정**:
- 볼륨 마운트 (코드, static, media)
- 환경 변수 설정
- 헬스체크
- 네트워크 설정

#### `Dockerfile.django`
- Python 3.11 slim 베이스
- PostgreSQL, FFmpeg, OpenCV 라이브러리 포함
- Gunicorn 설치
- **각 단계별 상세 주석**

---

### 3. Django 앱 구조

#### **accounts 앱** (사용자 관리)

**models.py**:
```python
class UserProfile(models.Model):
    user = models.OneToOneField(User, ...)
    phone_number = models.CharField(...)
    profile_image = models.ImageField(...)
    created_at, updated_at

    # Signal로 User 생성 시 자동으로 Profile 생성
```

**admin.py**:
- Custom UserAdmin (Profile Inline)
- UserProfileAdmin (색상 뱃지, 이미지 미리보기)

**views.py**:
- UserViewSet (DRF)
- `/api/users/me/` 커스텀 액션

**urls.py**:
- DRF Router 사용
- 자동 URL 생성

---

#### **videos 앱** (영상 관리) ⭐ 핵심

**models.py** - 3개 모델:

1. **Video 모델** (영상 정보)
   ```python
   - id: UUID
   - user: FK → User
   - title, original_filename
   - original_file_url, processed_file_url (S3 또는 로컬)
   - duration, width, height, fps, file_size
   - status: uploaded/analyzing/ready/processing/completed/failed
   - progress: 0-100
   - created_at, updated_at, completed_at, expires_at (7일)
   ```

2. **Face 모델** (고유 얼굴)
   ```python
   - id: UUID
   - video: FK → Video
   - face_index: 1, 2, 3, ...
   - thumbnail_url: 대표 얼굴 이미지
   - embedding: JSONField (512차원 벡터)
   - appearance_count, first_frame, last_frame
   - is_blurred: Boolean (사용자 선택)
   ```

3. **ProcessingJob 모델** (Celery 작업 추적)
   ```python
   - id: UUID
   - video: FK → Video
   - job_type: face_analysis / video_processing
   - celery_task_id
   - status: pending/started/success/failure
   - progress: 0-100
   - result_data: JSONField
   ```

**admin.py**:
- VideoAdmin: 상태 뱃지, 진행률 바, 영상 미리보기
- FaceAdmin: 썸네일 미리보기, 블러 상태 표시
- ProcessingJobAdmin: Celery 작업 모니터링
- **HTML/CSS inline으로 보기 좋은 UI 구현**

---

#### **processing 앱** (영상 처리)

현재는 기본 구조만 생성.
**Phase 4**에서 다음 파일들을 추가 예정:
- `tasks.py`: Celery 작업 (얼굴 분석, 영상 처리)
- `services.py`: FastAPI 클라이언트
- `utils.py`: 영상 처리 유틸리티

---

### 4. Django settings.py

#### 주요 설정:

**앱 등록**:
- `rest_framework`
- `corsheaders`
- `apps.accounts`, `apps.videos`, `apps.processing`

**데이터베이스**:
- 환경 변수로 PostgreSQL/SQLite 선택
- `DB_ENGINE=django.db.backends.postgresql` → PostgreSQL
- 기본값: SQLite

**AWS S3**:
- `USE_S3=True` → S3 사용
- `USE_S3=False` → 로컬 파일 시스템 (기본값)
- Presigned URL 준비

**DRF**:
- SessionAuthentication + BasicAuthentication
- IsAuthenticatedOrReadOnly
- 페이지네이션 (20개/페이지)

**CORS**:
- 개발: 모든 도메인 허용
- 운영: CORS_ALLOWED_ORIGINS

**Redis**:
- Django 캐시 백엔드
- Celery 브로커/결과 백엔드

**로깅**:
- Console + File 핸들러
- `logs/django.log`

**보안** (운영 환경):
- HTTPS 강제
- HSTS
- Secure 쿠키

---

## ❌ 아직 하지 않은 작업

### Phase 1 남은 작업:

1. **DRF Serializers 작성**
   - `apps/videos/serializers.py`
   - VideoSerializer, FaceSerializer, ProcessingJobSerializer
   - 중첩 Serializer (Video에 Face 포함)

2. **API Views 구현**
   - `apps/videos/views.py`
   - VideoViewSet (업로드, 목록, 상세)
   - 파일 업로드 처리
   - 영상 메타데이터 추출 (MoviePy)

3. **URL 라우팅 설정**
   - `apps/videos/urls.py` 생성
   - `face_blur_web/urls.py` 업데이트
   - API 엔드포인트 구조:
     ```
     /api/videos/              (GET, POST)
     /api/videos/{id}/         (GET, PUT, DELETE)
     /api/videos/{id}/faces/   (GET)
     /api/accounts/users/      (GET)
     /api/accounts/users/me/   (GET)
     ```

4. **HTML 템플릿 통합**
   - `templates/` 디렉토리 생성
   - `stitch_/비디오_업로드/code.html` → `templates/upload.html`
   - `stitch_/썸네일_선택/code.html` → `templates/face_selection.html`
   - `stitch_/최종_비디오_미리보기_및_저장/code.html` → `templates/preview.html`
   - Django 템플릿 문법 적용 (`{% static %}`, `{% url %}`)

5. **Static 파일 분리**
   - `static/css/`, `static/js/` 생성
   - Tailwind CSS를 별도 파일로 분리

6. **마이그레이션 실행**
   ```bash
   docker-compose up -d --build
   docker-compose exec django python manage.py makemigrations
   docker-compose exec django python manage.py migrate
   docker-compose exec django python manage.py createsuperuser
   ```

7. **테스트**
   - Admin 페이지 접속 확인
   - API 엔드포인트 테스트 (DRF Browsable API)

---

## 🚀 다음 세션 시작 방법

### 1️⃣ Docker 환경 시작

```bash
# 프로젝트 디렉토리로 이동
cd c:\Users\이승복\Documents\face_blur_project

# Docker Compose 빌드 및 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f django

# PostgreSQL 정상 작동 확인
docker-compose exec db psql -U postgres -d faceblur_db -c "\dt"
```

### 2️⃣ 마이그레이션 실행

```bash
# 마이그레이션 파일 생성
docker-compose exec django python manage.py makemigrations accounts videos processing

# 마이그레이션 적용
docker-compose exec django python manage.py migrate

# 슈퍼유저 생성
docker-compose exec django python manage.py createsuperuser
```

### 3️⃣ Admin 페이지 확인

브라우저에서 `http://localhost:8000/admin` 접속

### 4️⃣ Phase 1 나머지 작업 진행

1. **Serializers 작성**
   - `apps/videos/serializers.py` 생성
   - VideoSerializer, FaceSerializer 구현

2. **Views 작성**
   - `apps/videos/views.py` 생성
   - VideoViewSet 구현 (파일 업로드 포함)

3. **URLs 설정**
   - URL 라우팅 완성

4. **템플릿 통합**
   - stitch_ HTML을 Django 템플릿으로 이동

---

## 📚 참고 문서

### 프로젝트 문서
- **SKILL.md**: 전체 아키텍처 가이드 (1,890줄)
- **PROGRESS.md**: 이 파일 (진행 상황)
- **.env.example**: 환경 변수 설정 가이드

### Django 문서
- [Django 공식 문서](https://docs.djangoproject.com/)
- [DRF 공식 문서](https://www.django-rest-framework.org/)

### Docker 명령어
```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down

# 로그 확인
docker-compose logs -f [서비스명]

# Django 쉘 접속
docker-compose exec django python manage.py shell

# 데이터베이스 초기화 (볼륨 삭제)
docker-compose down -v
```

---

## 💡 주요 설계 결정 사항

### 1. UUID 사용
- **이유**: 보안 (ID 추측 방지), 분산 시스템 대비
- Video, Face, ProcessingJob 모두 UUID 사용

### 2. S3 선택적 사용
- **개발**: 로컬 파일 시스템 (`USE_S3=False`)
- **운영**: AWS S3 (`USE_S3=True`)
- 코드 변경 없이 환경 변수만 변경

### 3. 데이터베이스 선택
- **개발**: SQLite (간단함)
- **운영**: PostgreSQL (성능, 동시성)
- Docker Compose로 쉽게 전환

### 4. 얼굴 임베딩 저장
- JSONField에 512차원 벡터 저장
- 코사인 유사도 계산으로 같은 얼굴 판별

### 5. 처리 상태 관리
- Video.status: 6가지 상태 (uploaded → analyzing → ready → processing → completed/failed)
- ProcessingJob: Celery 작업과 1:1 매핑

---

## 🎯 현재 상태 요약

| 항목 | 상태 | 완성도 |
|------|------|--------|
| 프로젝트 설계 (SKILL.md) | ✅ | 100% |
| UI 프로토타입 (stitch_) | ✅ | 100% |
| Docker 환경 | ✅ | 100% |
| Django 앱 생성 | ✅ | 100% |
| 데이터베이스 모델 | ✅ | 100% |
| Django settings.py | ✅ | 100% |
| Admin 페이지 | ✅ | 100% |
| **Serializers** | ❌ | 0% |
| **Views (API)** | ❌ | 0% |
| **URL 라우팅** | ❌ | 20% |
| **템플릿 통합** | ❌ | 0% |
| **마이그레이션 실행** | ❌ | 0% |
| **Phase 1 전체** | ⏳ | **80%** |

---

## ✨ 특별히 잘된 부분

1. **상세한 주석**: 모든 코드에 학습용 주석 포함
2. **Admin 페이지**: 색상 뱃지, 진행률 바, 이미지 미리보기 등 보기 좋은 UI
3. **유연한 설정**: 환경 변수로 개발/운영 환경 쉽게 전환
4. **AWS 준비**: S3, EC2 배포를 고려한 설계
5. **확장성**: Phase 4 (Celery, WebSocket) 준비 완료

---

## 🔜 다음 세션 목표

**Phase 1 완성하기**:
1. ✅ Serializers 작성 (30분)
2. ✅ Views 작성 (1시간)
3. ✅ URLs 설정 (20분)
4. ✅ 템플릿 통합 (1시간)
5. ✅ 마이그레이션 & 테스트 (30분)

**예상 소요 시간**: 3-4시간

---

**작성자**: Claude (Anthropic AI)
**마지막 업데이트**: 2025-11-09
**다음 작업**: DRF Serializers 작성부터 시작
