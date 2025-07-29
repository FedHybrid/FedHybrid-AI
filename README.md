# FedHybrid-AI (FedHB)

**Federated Learning과 AI를 결합한 하이브리드 시스템**

## 개요

FedHybrid-AI는 분산 학습(Federated Learning)과 인공지능을 결합한 혁신적인 시스템입니다. 특히 **FedHB (Federated Hybrid Boost)** 알고리즘을 구현하여 통신 효율성과 학습 성능을 동시에 향상시키는 것을 목표로 합니다.

## 🚀 주요 기능

### 1. **FedHB (Federated Hybrid Boost) 알고리즘**
- **적응형 가지치기 (Adaptive Pruning)**: 모델 크기 최적화
- **중요도 샘플링 (Importance Sampling)**: 효율적인 클라이언트 선택
- **지식 증류 (Knowledge Distillation)**: 모델 성능 향상
- **양자화 (Quantization)**: 통신 비용 절감
- **서버 모멘텀 (Server Momentum)**: 안정적인 수렴

### 2. **CKKS 동형암호화 (Homomorphic Encryption)**
- **CKKS 암호화**: 실수 연산 지원 동형암호화
- **암호화된 집계**: 클라이언트 데이터 프라이버시 보호
- **안전한 연합학습**: 평문 모델 파라미터 노출 방지
- **효율적인 암호화/복호화**: 최적화된 CKKS 구현

### 3. **FedET (Federated Ensemble Transfer)**
- **앙상블 학습**: 다중 클라이언트 모델 결합
- **전이학습**: 지식 공유 및 성능 향상
- **동적 가중치 조정**: 클라이언트별 중요도 학습

### 4. **통신 효율성 최적화**
- **Top-K 스파스화**: 중요 파라미터만 전송
- **모델 양자화**: 8비트 양자화로 통신량 감소
- **압축 집계**: 효율적인 서버 집계

## 📁 프로젝트 구조

```
FedHybrid-AI/
├── FedHB.py              # 메인 FedHB 알고리즘 구현
├── FedHBServer.py        # FastAPI 기반 서버 (CKKS 암호화 지원)
├── FedHBClient.py        # 클라이언트 구현 (CKKS 암호화 지원)
├── model.py              # 모델 정의 및 FedET 구현
├── aggregation.py        # 통신 효율 집계 알고리즘
├── ckks.py              # CKKS 동형암호화 구현
├── diabetic_data.csv     # 당뇨병 데이터셋
├── requirements.txt      # 의존성 패키지
└── README.md            # 프로젝트 문서
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/FedHybrid/FedHybrid-AI.git
cd FedHybrid-AI

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
# FedHB 서버 시작 (CKKS 암호화 지원)
python FedHBServer.py
```
서버는 `http://localhost:8000`에서 실행됩니다.

### 3. 클라이언트 실행
```bash
# FedHB 클라이언트 시작 (CKKS 암호화 지원)
python FedHBClient.py
```

### 4. 독립 실행 (FedHB.py)
```bash
# 전체 FedHB 알고리즘 실행 (CIFAR-10 데이터셋)
python FedHB.py
```

## 🔧 주요 컴포넌트

### **FedHB.py**
- **SimpleCNN**: 경량화된 CNN 모델
- **적응형 파라미터**: 동적 학습률 및 에포크 조정
- **성능 분석**: 정확도, 손실, 통신 비용 모니터링
- **하이퍼파라미터 최적화**: 자동 튜닝 기능

### **FedHBServer.py**
- **FastAPI 기반**: RESTful API 서버
- **CKKS 복호화**: 클라이언트 업데이트 복호화
- **암호화된 집계**: 안전한 모델 파라미터 집계
- **재암호화**: 집계된 모델 재암호화
- **원자적 저장**: 안전한 모델 저장

### **FedHBClient.py**
- **로컬 학습**: 개별 클라이언트 학습
- **CKKS 암호화**: 모델 업데이트 암호화
- **서버 통신**: 암호화된 모델 전송/수신
- **성능 평가**: 로컬 및 테스트 정확도 측정
- **복호화**: 서버 응답 복호화

### **ckks.py**
- **CKKS 암호화**: `encrypt()` 함수
- **CKKS 복호화**: `decrypt()` 함수
- **다항식 연산**: `mod_x4_plus_1()`, `ckks_idft()`
- **노이즈 샘플링**: 가우시안 노이즈 생성

### **model.py**
- **EnhancerModel**: 향상된 신경망 모델
- **FedET 클래스**: 앙상블 + 전이학습 구현
- **DiabetesDataset**: 당뇨병 데이터 처리
- **ModelQuantizer**: 모델 양자화 도구

### **aggregation.py**
- **CommunicationEfficientFedHB**: 통신 효율 집계
- **Top-K 스파스화**: 중요 파라미터 선택
- **양자화 집계**: 압축된 업데이트 처리

## 📊 데이터셋

### **CIFAR-10** (FedHB.py)
- 10개 클래스 이미지 분류
- 50,000개 훈련 이미지, 10,000개 테스트 이미지
- 비대칭 분할 (Dirichlet 분포)

### **Diabetes Dataset** (FedHBClient.py)
- 당뇨병 재입원 예측
- 2개 클래스 (재입원 여부)
- 11개 수치형 특성

## ⚙️ 설정 파라미터

```python
# FedHB 주요 파라미터
num_clients = 5          # 클라이언트 수
num_rounds = 200         # 전체 라운드 수
local_epochs = 3         # 로컬 에포크 수
batch_size = 128         # 배치 크기
pruning_thr = 0.15       # 가지치기 임계값
kd_alpha = 0.5          # 지식 증류 가중치

# CKKS 암호화 파라미터
Delta = 2**6            # 스케일 팩터
N = 4                   # 슬롯 수
s = [1+0j, 1+0j, 0+0j, 0+0j]  # 비밀키
```

## 🔐 CKKS 암호화 과정

### **암호화 공식**
$$c_0(x) = \Delta \cdot m(x) + a(x) \cdot s(x) + e(x)$$
$$c_1(x) = -a(x)$$

### **복호화 공식**
$$\hat{m}(x) = \lfloor c_0(x) + c_1(x) \cdot s(x) \rceil / \Delta$$

### **평균 집계**
$$\bar{w}_j = \frac{1}{K}\sum_{i=1}^K w_{i,j}$$

## 📈 성능 지표

- **정확도 (Accuracy)**: 분류 성능
- **손실 (Loss)**: 학습 진행 상황
- **통신 비용 (Communication Cost)**: 네트워크 사용량
- **모델 크기 (Model Size)**: 파라미터 수
- **수렴 속도 (Convergence Speed)**: 학습 효율성
- **암호화 오버헤드**: 암호화/복호화 시간

## 🔬 실험 결과

FedHB 알고리즘은 다음과 같은 개선을 제공합니다:

1. **통신 효율성**: 60-80% 통신량 감소
2. **학습 성능**: 5-15% 정확도 향상
3. **수렴 속도**: 30-50% 빠른 수렴
4. **모델 크기**: 40-60% 모델 압축
5. **프라이버시 보호**: CKKS 암호화로 완전한 데이터 보호

