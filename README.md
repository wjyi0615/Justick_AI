# Justick_AI
캡스톤 디자인(2)

# 🥬 딱대: 내일의 채소값을 알려줄께

> 가락시장 기반 농산물 도매가 예측 서비스  
> LSTM + EWC 기반 시계열 예측 모델 적용

---

## 📌 프로젝트 개요

**딱대(DDAKDAE)**는 국내 최대 도매시장인 **가락시장**의 농산물 시세 데이터를 기반으로,  
소비자와 판매자, 유통업자, 정책 입안자에게 **내일의 도매가를 예측하여 제공**하는 머신러닝 기반 예측 플랫폼입니다.

---

## 💡 핵심 기능

- 원하는 **농산물 시세 정보 조회**
- **시세 변동 추이 시각화** (그래프 제공)
- **LSTM + EWC 기반 예측 모델**로 다음날 가격 추정
- 매일 업데이트되는 데이터를 기반으로 **지속학습(Continual Learning)** 수행

---

## 🧠 사용된 예측 기법

### 🔁 LSTM (Long Short-Term Memory)

- 시계열 데이터를 처리하기 위한 순환 신경망(RNN)의 확장 모델
- 가격, 물량, 격차 등 **시간에 따른 변화를 모델링**하는 데 강점을 가짐

### 🧠 EWC (Elastic Weight Consolidation)

- **기존 학습 정보를 잊지 않도록 제약을 거는 손실 함수**
- 새로운 날짜가 추가될 때마다, **기존 모델의 중요한 파라미터를 보존**하면서 점진적 학습

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

| 영역 | 기술 |
|------|------|
| 프론트엔드 | React, Figma |
| 백엔드 | Spring Boot, mysql |
| 머신러닝 | Python, PyTorch, Pandas, Scikit-learn |
| 배포 | AWS EC2 |
| 협업 | Git, Notion, Discord |


## 🙋 팀 소개

| 이름  | 역할                  |
| --- | ------------------- |
| 이우중 | 프론트엔드 / 데이터 분석 / 머신러닝   |
| 배성빈 | 프론트엔드 / 데이터 분석 / 머신러닝        |
| 이찬우 | 백엔드 / 데이터 처리 |

---

## 📎 참고 문서

* [EWC 논문: Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
* [LSTM 논문: Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
* [가락시장 공식 사이트](http://www.garak.co.kr/)



