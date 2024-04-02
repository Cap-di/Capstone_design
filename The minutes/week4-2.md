## 4-2주차 회의록

### 데이터

데이터 구조, 활용 방안 제출

---

### [transformer](../TriplaneGaussian/transformers.py) 코드 분석

#### 기본 클래스와 유틸리티 함수 - 조시현

-   MemoryEfficientAttentionMixin
-   GatedSelfAttentionDense
-   FeedForward, GELU, GEGLU, ApproximateGELU

#### Attention 레이어 - 차성철

-   Attention

#### Normalization 레이어 - 김대형

-   AdaLayerNorm
-   daLayerNormContinuous
-   Modulation
-   AdaLayerNormZero
-   AdaGroupNorm

#### 기본 Transformer 블록 - 유광열

-   BasicTransformerBlock

#### 1D Transformer 모델 - 허연후

-   Transformer1D

---

### 각자 알아야 할 것

[TriplaneGaussian 논문](https://arxiv.org/pdf/2312.09147.pdf) 공부해 오기
