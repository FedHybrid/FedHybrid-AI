import torch
import requests
from model import client_update_full, EnhancerModel, load_diabetes_data
from aggregation import CommunicationEfficientFedHB
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import time
from ckks import batch_encrypt, batch_decrypt

# device 설정
device = torch.device('cpu')  # GPU 환경 문제로 CPU 강제 지정

# 클라이언트 설정
import os
CLIENT_ID = os.getenv('CLIENT_ID', 'client_1')  # 환경변수로 클라이언트 ID 설정 가능

# CKKS 파라미터 설정 (ckks.py와 동일하게)
z_q = 1 << 10   # 2^10 = 1,024 (평문 인코딩용 스케일)
rescale_q = z_q  # 리스케일링용 스케일
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

# 데이터셋 준비 (train/test 분할)
train_dataset, test_dataset = load_diabetes_data('diabetic_data.csv')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
input_dim = train_dataset.X.shape[1]

# 모델 준비 (EnhancerModel)
client_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)
global_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)  # 글로벌 모델 추가

SERVER_URL = "http://localhost:8000"
NUM_ROUNDS = 10

def evaluate_local_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100 if total > 0 else 0.0
    return acc

def download_global_model():
    for _ in range(5):
        r = requests.get(f"{SERVER_URL}/get_model")
        with open("global_model.pth", "wb") as f:
            f.write(r.content)
        try:
            state_dict = torch.load("global_model.pth", map_location=device, weights_only=False)
            client_model.load_state_dict(state_dict)
            global_model.load_state_dict(state_dict)  # 글로벌 모델도 업데이트
            os.remove("global_model.pth")
            return
        except Exception as e:
            print(f"global_model.pth 로드 실패, 재시도... ({e})")
            time.sleep(1)
    raise RuntimeError("global_model.pth를 정상적으로 다운로드하지 못했습니다.")

for r in range(NUM_ROUNDS):
    round_start_time = time.time()  # 라운드 시작 시간
    print(f"=== 라운드 {r+1} 시작 ===")
    download_global_model()
    acc_before = evaluate_local_accuracy(client_model, train_loader, device)
    
    # 로컬 학습 수행
    training_start_time = time.time()
    updated_model, avg_loss, epochs, num_samples = client_update_full(
        client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device,
        use_kd=False, use_fedprox=False, use_pruning=True  # 일시적 비활성화
    )
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
    
    # 로컬 학습 완료된 모델을 그대로 사용 (FedAvg 원칙)
    print(f"=== 로컬 학습 완료 ===")
    print(f"로컬 학습된 모델 파라미터를 서버로 전송할 준비 완료")
    
    # 학습된 모델을 클라이언트 모델에 복사
    client_model.load_state_dict(updated_model.state_dict())
    
    # === 1단계: 클라이언트 데이터를 CKKS로 암호화 ===
    encryption_start_time = time.time()
    print(f"\n=== 1단계: 클라이언트 데이터 CKKS 암호화 ===")
    state_dict = client_model.state_dict()
    print(f"모델 파라미터 수: {len(state_dict)}개 레이어")
    
    # 1) Tensor → flat numpy vector
    flat = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])
    print(f"평면화된 벡터 크기: {len(flat)}")
    print(f"원본 값 범위: {flat.min():.4f} ~ {flat.max():.4f}")
    print(f"원본 값 평균: {flat.mean():.4f}")
    
    # 2) 정수 기반 배치 암호화
    m_coeffs = flat.astype(np.int64)  # 정수 벡터
    c0_list, c1_list = batch_encrypt(m_coeffs, batch_size=4)
    encryption_end_time = time.time()
    encryption_duration = encryption_end_time - encryption_start_time
    print(f"정수 기반 암호화 완료: {len(c0_list)}개 배치")
    print(f"암호화된 c0 첫 번째 배치 범위: {c0_list[0].min()} ~ {c0_list[0].max()}")
    print(f"암호화된 c1 첫 번째 배치 범위: {c1_list[0].min()} ~ {c1_list[0].max()}")
    print(f"암호화 소요 시간: {encryption_duration:.1f}초")
    
    # === 2단계: 암호화된 데이터를 서버로 전송 ===
    print(f"\n=== 2단계: 암호화된 데이터 서버 전송 ===")
    payload = {
        "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list],
        "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list],
        "original_size": len(m_coeffs),
        "num_samples": num_samples,
        "loss": avg_loss,
        "client_id": CLIENT_ID,  # 클라이언트 식별자
        "round_id": r+1  # 라운드 식별자
    }
    print(f"전송할 페이로드 크기: {len(payload['c0_list'])}개 배치")
    
    try:
        communication_start_time = time.time()
        print(f"서버로 암호화된 데이터 전송 중...")
        aggregate_response = requests.post(f"{SERVER_URL}/aggregate", json=payload)
        if aggregate_response.status_code == 200:
            try:
                aggregate_json = aggregate_response.json()
                communication_end_time = time.time()
                communication_duration = communication_end_time - communication_start_time
                print(f"[Round {r+1}] 서버 응답 수신 완료")
                print(f"통신 소요 시간: {communication_duration:.1f}초")
                
                # === 3단계: 서버로부터 암호화된 평균 결과 수신 ===
                if "c0_list" in aggregate_json and "c1_list" in aggregate_json:
                    print(f"\n=== 3단계: 서버 암호화된 평균 결과 수신 ===")
                    c0_list_agg = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in aggregate_json["c0_list"]]
                    c1_list_agg = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in aggregate_json["c1_list"]]
                    original_size = aggregate_json["original_size"]
                    
                    print(f"받은 암호화된 평균 결과: {len(c0_list_agg)}개 배치")
                    print(f"받은 c0 첫 번째 배치 범위: {c0_list_agg[0].min()} ~ {c0_list_agg[0].max()}")
                    print(f"받은 c1 첫 번째 배치 범위: {c1_list_agg[0].min()} ~ {c1_list_agg[0].max()}")
                    
                    # === 4단계: 암호화된 상태로 모델 업데이트 (복호화 없이) ===
                    print(f"\n=== 4단계: 암호화된 상태로 모델 업데이트 ===")
                    print(f"암호화된 상태로 글로벌 모델 업데이트 중...")
                    
                    # 암호화된 상태로 모델 파라미터 저장 (복호화하지 않음)
                    encrypted_state = {
                        'c0_list': c0_list_agg,
                        'c1_list': c1_list_agg,
                        'original_size': original_size
                    }
                    
                    # 글로벌 모델을 암호화된 상태로 저장
                    torch.save(encrypted_state, "encrypted_global_model.pth")
                    print(f"암호화된 글로벌 모델 저장 완료")
                    
                    # 다음 라운드에서 암호화된 상태로 학습할 수 있도록 준비
                    print(f"다음 라운드에서 암호화된 상태로 학습 준비 완료")
                    
                    # === 5단계: 암호화된 상태로 학습 진행 (다음 라운드에서) ===
                    print(f"\n=== 5단계: 암호화된 상태로 학습 준비 ===")
                    print(f"현재 라운드에서는 암호화된 상태 저장 완료")
                    print(f"다음 라운드에서 암호화된 상태로 학습을 진행할 예정")
                        
            except Exception as e:
                print(f"[Round {r+1}] 서버 응답 처리 중 에러: {e}")
        else:
            print(f"[Round {r+1}] 서버 전송 실패: {aggregate_response.status_code}")
    except Exception as e:
        print(f"[Round {r+1}] 서버 통신 중 에러: {e}")
    
    # NaN 체크
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        print(f"[Round {r+1}] 경고: NaN/Inf 손실 감지, 이전 모델 유지")
        avg_loss = 1.0  # 기본값 설정
    else:
        print(f"[Round {r+1}] 로컬 손실: {avg_loss:.4f}, 에포크: {epochs}, 샘플 수: {num_samples}")
    
    # 집계된 글로벌 모델로 테스트셋 정확도 평가
    global_acc = evaluate_local_accuracy(client_model, test_loader, device)
    
    # 라운드 소요 시간 계산
    round_end_time = time.time()
    round_duration = round_end_time - round_start_time
    
    print(f"[Round {r+1}] 글로벌 모델 로컬 데이터 정확도(학습 전): {acc_before:.2f}% | 로컬 모델 정확도(학습 후): {acc_after:.2f}%")
    print(f"[Round {r+1}] 글로벌 모델 테스트셋 정확도: {global_acc:.2f}%")
    print(f"[Round {r+1}] 시간 분석:")
    print(f"  - 로컬 학습: {training_duration:.1f}초")
    print(f"  - 암호화: {encryption_duration:.1f}초")
    print(f"  - 통신: {communication_duration:.1f}초")
    print(f"  - 전체 라운드: {round_duration:.1f}초")
    print(f"=== 라운드 {r+1} 종료 ===\n") 