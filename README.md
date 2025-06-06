# Justick_AI
캡스톤 디자인(2)

## 🌾 농산물 가격 예측 시스템

**작물별 시계열 특성에 따라 최적화된 모델을 자동 선택하고,  
단기 추세와 계절성까지 반영하는 지능형 농산물 예측 플랫폼입니다.**

---

## 📌 프로젝트 개요

**딱대(Justick)** 는 국내 최대 도매시장인 **가락시장**의 농산물 시세 데이터를 기반으로,  
소비자와 판매자, 유통업자, 정책 입안자에게 **내일의 도매가를 예측하여 제공**하는 머신러닝 기반 예측 플랫폼입니다.

---

## 💡 핵심 기능

- 원하는 **농산물 도매 시세 정보 조회**  
- **과거 시세 변동 추이 시각화** (월별 그래프 제공)
- 작물에 따라 DLinear, XGBoost, **LSTM+EWC, DoubleAdapt** 등 **맞춤형 모델 자동 적용**
- **LSTM + EWC 기반 지속학습(Continual Learning)** 으로 **매일 업데이트되는 데이터를 반영**
- **1일, 7일, 28일 단위 예측** 결과 제공
---

## 🧠 사용된 예측 기법

### 🔁 LSTM (Long Short-Term Memory)

- 시계열 데이터를 처리하기 위한 순환 신경망(RNN)의 확장 모델
- 가격, 물량, 격차 등 **시간에 따른 변화를 모델링**하는 데 강점을 가짐

### 🧠 EWC (Elastic Weight Consolidation)

- **기존 학습 정보를 잊지 않도록 제약을 거는 손실 함수**
- 새로운 날짜가 추가될 때마다, **기존 모델의 중요한 파라미터를 보존**하면서 점진적 학습

### ⚡ DLinear (Decomposition Linear Model)
- 시계열 데이터를 **trend + seasonal 성분으로 분해** 후 각각 선형 예측
- **패턴이 명확하고 반복적인 작물에 경량/고속 처리에 적합**

### 🌱 DoubleAdapt (Meta Learning 기반)
- **데이터 수가 적고 불확실성이 높은 작물**을 위한 메타러닝 기반 적응 예측
- 작물마다 모델을 **few-shot 방식으로 빠르게 적응**
---

## 📊 사용 데이터 및 피처

- 데이터 출처: 가락시장 농산물 도매가
- 사용 대상: 양파, 배추, 무, 감자, 고구마, 토마토(HIGH,Speacial 등급)
- 주요 입력 피처:
  - intake (반입량)
  - gap (전일대비 반입 격차)
  - price_diff (이전일 대비 가격 차이)
  - rolling_mean / rolling_std (3일 이동 평균 및 표준편차)
  - sin_day / cos_day (연중 주기 특성)

---

## 🛠 기술 스택

| 영역        | 기술                                    |
| --------- | ------------------------------------- |
| **프론트엔드** | React, Figma                          |
| **백엔드**   | Spring Boot, MySQL                    |
| **머신러닝**  | Python, PyTorch, Pandas, Scikit-learn |
| **배포**    | AWS EC2, Docker, Kubernetes   |


## 🙋 팀 소개

| 이름  | 역할                  |
| --- | ------------------- |
| 이우중 | 데이터 분석 / 머신러닝   |
| 배성빈 | 프론트엔드 / 머신러닝        |
| 이찬우 | 백엔드 / 데이터 처리 |

---

## 📚 참고 논문 / 기술 기반

- **EWC**  
  *Overcoming catastrophic forgetting in neural networks*, Kirkpatrick et al., 2017 (PNAS)  
  👉 [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

- **DLinear**  
  *Time Series is a Special Sequence: Forecasting with Decomposition and Graph Neural Networks*, Zeng et al., NeurIPS 2022  
  👉 [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

- **DoubleAdapt**  
  *Few-shot Adaptive Time Series Forecasting via Meta-Learning*, Zhang et al., ICLR 2023  
  👉 [arXiv:2210.10088](https://arxiv.org/abs/2210.10088)

- **XGBoost**  
  *XGBoost: A Scalable Tree Boosting System*, Chen & Guestrin, KDD 2016  
  👉 [ACM Digital Library](https://dl.acm.org/doi/10.1145/2939672.2939785)

- [Attention is All You Need (Self-Attention)](https://arxiv.org/abs/1706.03762)  
- [가락시장 공식 사이트 (데이터 출처)](http://www.garak.co.kr/)




