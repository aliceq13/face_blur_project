# 얼굴 인식 프로젝트 - 심각한 허점 분석 보고서

## 🚨 치명적인 문제점들

### 1. **임베딩 품질 문제**

#### 문제 1-1: AdaFace 임베딩이 정규화되지 않음
**위치**: `apps/videos/adaface_wrapper.py:82`
```python
return embedding.cpu().numpy()[0]  # ❌ 정규화 안 됨!
```

**영향**:
- AdaFace 모델 출력이 정규화되지 않은 상태로 반환됨
- Cosine similarity 계산 시 부정확한 결과
- 클러스터링 성능 저하

**해결책**:
```python
embedding = embedding.cpu().numpy()[0]
# L2 정규화 필수
embedding = embedding / np.linalg.norm(embedding)
return embedding
```

---

#### 문제 1-2: YOLO Tight Crop으로 인한 얼굴 정렬 부재
**위치**: `apps/videos/face_detection.py:269-271`
```python
face_img_bgr = frame[y1:y2, x1:x2]  # ❌ Tight crop만 사용
face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
```

**영향**:
- 얼굴이 기울어진 경우 임베딩 품질 저하
- 옆모습, 위/아래 각도에서 인식률 급감
- AdaFace/ArcFace는 정렬된 얼굴을 기대함

**해결책**:
- 5-point landmark 기반 얼굴 정렬 추가
- 또는 InsightFace의 alignment 사용

---

### 2. **클러스터링 알고리즘 문제**

#### 문제 2-1: HDBSCAN이 Faiss를 전혀 활용하지 않음
**위치**: `apps/videos/face_detection.py:403-428`
```python
# Faiss HNSW 인덱스 생성 (403-408)
index = faiss.IndexHNSWFlat(d, 32)
index.add(embeddings_array)

# k-NN 검색 (410-413)
D, I = index.search(embeddings_array, k_neighbors)  # ❌ 사용 안 함!

# HDBSCAN은 embeddings를 직접 사용 (428)
labels = clusterer.fit_predict(embeddings_array)  # ❌ Faiss 결과 무시
```

**영향**:
- Faiss 인덱스를 만들고도 전혀 사용하지 않음
- 불필요한 연산 낭비
- 사용자가 제공한 코드의 의도를 완전히 무시

**해결책**:
- Faiss를 사용하지 않거나
- k-NN 그래프를 HDBSCAN에 제대로 전달

---

#### 문제 2-2: 동적 파라미터 조정이 너무 보수적
**위치**: `apps/videos/face_detection.py:417-418`
```python
min_cluster_size = max(2, min(5, len(embeddings_array) // 3))  # ❌ 너무 작음
min_samples_val = max(1, min(5, len(embeddings_array) // 5))   # ❌ 너무 작음
```

**영향**:
- 8개 tracklet → min_cluster_size=2, min_samples=1
- 거의 모든 것을 하나의 클러스터로 묶음
- 다른 사람도 같은 사람으로 인식

**해결책**:
```python
# 사용자가 제공한 원래 값 사용
min_cluster_size = 5
min_samples = 5
```

---

### 3. **Tracklet 생성 문제**

#### 문제 3-1: ByteTrack의 불안정성
**위치**: `apps/videos/face_detection.py:234-250`
```python
results = self.yolo_model.track(
    source=video_path,
    tracker="botsort.yaml",  # ❌ ByteTrack 대신 BotSORT 사용
    ...
)
```

**영향**:
- Track ID가 자주 바뀜
- 같은 사람이 여러 tracklet으로 분리됨
- 클러스터링 전에 이미 문제 발생

---

#### 문제 3-2: 선명도 계산이 부정확
**위치**: `apps/videos/face_detection.py:180-188`
```python
def _calculate_clarity(self, img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()  # ❌ Laplacian만 사용
```

**영향**:
- 노이즈가 많은 이미지가 "선명"하다고 판단됨
- 실제로 흐릿한 얼굴이 선택될 수 있음

**해결책**:
- Gaussian blur + Laplacian 조합
- 또는 FFT 기반 선명도 측정

---

### 4. **Re-ID (재식별) 문제**

#### 문제 4-1: vote_rate가 너무 낮음
**위치**: `apps/videos/video_blurring.py:235`
```python
if vote_rate > 0.4:  # ❌ 40%만 일치해도 동일 인물
```

**영향**:
- 다른 사람도 쉽게 같은 사람으로 인식
- False positive 증가

---

#### 문제 4-2: similarity threshold가 너무 높음
**위치**: `apps/videos/video_blurring.py` (Re-ID 로직)
```python
max_sim > 0.9  # ❌ 너무 엄격함
```

**영향**:
- 같은 사람도 다른 사람으로 인식
- 각도/조명 변화에 취약

---

### 5. **아키텍처 설계 문제**

#### 문제 5-1: 4M 모델 사용
**현재**: `adaface_vit_base_kprpe_webface4m.pt`
**문제**: 4M은 작은 데이터셋으로 학습됨

**해결**: 12M 모델 사용 (이미 코드 수정됨, 모델만 다운로드 필요)

---

#### 문제 5-2: 썸네일 1개만 저장
**위치**: `apps/videos/face_detection.py:507-510`
```python
# 썸네일 저장 (가장 선명한 1장)  # ❌ 1개만
thumbnail_filename = f"face_{face_index}.jpg"
```

**영향**:
- 다양한 각도의 얼굴을 대표하지 못함
- Re-ID 시 각도 변화에 취약

---

## 📊 우선순위별 수정 권장사항

### 🔴 긴급 (즉시 수정 필요)

1. **AdaFace 임베딩 정규화** - 가장 치명적
2. **HDBSCAN 파라미터 복원** - min_cluster_size=5, min_samples=5
3. **Faiss 사용 로직 제거 또는 제대로 구현**

### 🟡 중요 (성능 개선)

4. **얼굴 정렬 추가** - 5-point landmark
5. **12M 모델로 업그레이드**
6. **Re-ID threshold 조정** - vote_rate=0.5, sim_threshold=0.85

### 🟢 개선 (추가 최적화)

7. **선명도 계산 개선** - Gaussian + Laplacian
8. **썸네일 3개 저장** - 다양한 각도
9. **ByteTrack → DeepSORT 변경**

---

## 🎯 결론

**가장 큰 문제**: 
1. **임베딩이 정규화되지 않음** → Cosine similarity 계산이 무의미
2. **HDBSCAN 파라미터가 너무 약함** → 모든 것을 하나로 묶음
3. **Faiss를 만들고도 사용하지 않음** → 불필요한 연산

이 3가지만 수정해도 성능이 크게 개선될 것입니다.
