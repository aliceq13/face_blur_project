# 임베딩 벡터 비교 방법론 및 개선 계획

현재 프로젝트는 **Cosine Similarity**를 기반으로 얼굴 임베딩을 비교하고 있습니다. 이는 얼굴 인식 분야의 표준이지만, 더 높은 정확도와 강건함(Robustness)을 위해 적용할 수 있는 다양한 방법론들이 존재합니다.

## 1. 현재 방식 분석
- **방식**: Cosine Similarity (L2 정규화된 벡터의 내적)
- **장점**: 계산이 빠르고, AdaFace/ArcFace 등 최신 모델의 학습 목표(Angular Margin)와 일치합니다.
- **한계**: 
  - 조명, 각도, 가림 등으로 인한 "Hard Positive" (같은 사람인데 다르게 보이는 경우)를 놓칠 수 있음.
  - "Hard Negative" (다른 사람인데 비슷하게 보이는 경우)를 잘못 매칭할 수 있음.
  - 단순 1:1 비교는 주변 데이터 분포(Context)를 고려하지 않음.

## 2. 적용 가능한 방법론 (Methodologies)

### A. 거리 측정 지표 (Metric-based)
1.  **Euclidean Distance (L2)**:
    - Cosine Similarity와 수학적으로 밀접한 관계 ($L2^2 = 2(1 - Cosine)$).
    - 순위(Ranking)는 동일하지만, 거리 기반 클러스터링(HDBSCAN 등) 라이브러리들이 기본적으로 지원하므로 호환성이 좋습니다.
2.  **Mahalanobis Distance**:
    - 데이터의 분산(공분산)을 고려한 거리.
    - 특정 차원의 변동성이 크다면 그 차원의 가중치를 줄여줍니다. (계산 비용 높음)

### B. 집합 기반 비교 (Set-based Comparison)
*단일 평균 임베딩(Average Embedding) 대신 여러 장의 이미지를 집합으로 비교*
1.  **Hausdorff Distance**: 두 집합 간의 가장 먼 거리의 최소값 등을 이용.
2.  **Mean-Set Similarity**: 집합 간의 모든 쌍(Pair)의 유사도 평균.
3.  **Top-k Average**: 각 집합에서 가장 유사한 k개의 쌍의 평균을 사용. (이상치에 강함)

### C. 고급 Re-ID 기법 (Advanced Techniques) - **추천** 🌟
1.  **Query Expansion (QE)**:
    - 쿼리 이미지와 가장 유사한 Top-k 이미지를 찾아, 그들의 임베딩을 평균내어 "새로운 쿼리"로 사용.
    - **효과**: 노이즈가 제거되고 더 강건한 쿼리 생성 가능.
2.  **Database-side Feature Augmentation (DBA)**:
    - 갤러리(저장된 얼굴들) 쪽에서도 유사한 이웃들의 특징을 결합하여 갱신.
3.  **k-Reciprocal Re-ranking**:
    - A가 B를 Top-1으로 뽑았을 때, B도 A를 Top-1으로 뽑는지 확인 (상호 일치).
    - 상호 일치하는 이웃(k-reciprocal nearest neighbors) 정보를 이용하여 유사도 점수를 재조정(Re-ranking).
    - **효과**: **정확도(mAP)를 비약적으로 상승**시키는 검증된 기법.

### D. 품질 기반 가중치 (Quality-Aware)
1.  **Face Quality Weighting**:
    - 선명도(Clarity), 정면 여부(Pose) 등을 점수화하여, 품질이 좋은 얼굴의 임베딩에 더 높은 가중치를 부여.
    - 현재 `TW-FINCH` 구현에 일부 적용되어 있으나, Re-ID 단계에서도 적용 가능.

## 3. 제안하는 개선 계획 (Implementation Plan)

### 단계 1: 품질 기반 매칭 (Quality-Aware Matching) 도입
- **내용**: Re-ID 시 단순 Cosine Similarity 대신, 선명도가 높은 썸네일(Anchor)과의 유사도에 가중치를 부여.
- **이유**: 흐릿한 얼굴끼리의 유사도는 신뢰할 수 없음.

### 단계 2: Query Expansion (QE) 구현
- **내용**: Re-ID 검색 시, 1차 검색 결과에서 매우 확실한(유사도 0.8 이상) 얼굴들을 찾아 쿼리 벡터를 보정(Update)한 후 2차 검색 수행.
- **이유**: 각도/조명이 달라도 중간 단계의 얼굴을 통해 연결될 수 있음.

### 단계 3: k-Reciprocal Re-ranking (선택적)
- **내용**: 계산 비용은 조금 들지만, 정확도가 매우 중요한 경우 적용.
- **이유**: "닮은 꼴" 오인식을 획기적으로 줄일 수 있음.

## 4. 사용자 리뷰 요청
- 위 방법론 중 **어떤 방향**으로 진행할까요?
- **추천**: **단계 1 (품질 기반)**과 **단계 2 (Query Expansion)**를 우선 적용하는 것이 가성비(속도 vs 정확도)가 가장 좋습니다.
