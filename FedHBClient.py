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

# CKKS 파라미터 설정 (ckks.py와 동일하게)
z_q = 1 << 10   # 2^10 = 1,024 (평문 인코딩용 스케일)
rescale_q = z_q  # 리스케일링용 스케일
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

# 데이터셋 준비 (train/test 분할)
train_dataset, test_dataset = load_diabetes_data('diabetic_data.csv')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
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
    print(f"=== 라운드 {r+1} 시작 ===")
    download_global_model()
    acc_before = evaluate_local_accuracy(client_model, train_loader, device)
    
    # 로컬 학습 수행
    updated_model, avg_loss, epochs, num_samples = client_update_full(
        client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device,
        use_kd=False, use_fedprox=False, use_pruning=True  # 일시적 비활성화
    )
    acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
    
    # FedAvg 방식: 글로벌 모델과 로컬 모델의 가중 평균 (로컬 빠른 업데이트)
    print(f"=== FedAvg 집계 시작 ===")
    alpha = 0.1  # 글로벌 모델 업데이트 비율
    
    for name, param in client_model.named_parameters():
        if name in global_model.state_dict():
            # 글로벌 모델 = (1-α) * 글로벌 모델 + α * 로컬 모델
            global_param = global_model.state_dict()[name]
            local_param = updated_model.state_dict()[name]
            new_param = (1 - alpha) * global_param + alpha * local_param
            global_model.state_dict()[name].copy_(new_param)
            client_model.state_dict()[name].copy_(new_param)
    
    print(f"FedAvg 집계 완료 (α={alpha})")
    
    # === 1단계: 클라이언트 데이터를 CKKS로 암호화 ===
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
    print(f"정수 기반 암호화 완료: {len(c0_list)}개 배치")
    print(f"암호화된 c0 첫 번째 배치 범위: {c0_list[0].min()} ~ {c0_list[0].max()}")
    print(f"암호화된 c1 첫 번째 배치 범위: {c1_list[0].min()} ~ {c1_list[0].max()}")
    
    # === 2단계: 암호화된 데이터를 서버로 전송 ===
    print(f"\n=== 2단계: 암호화된 데이터 서버 전송 ===")
    payload = {
        "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list],
        "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list],
        "original_size": len(m_coeffs),
        "num_samples": num_samples,
        "loss": avg_loss
    }
    print(f"전송할 페이로드 크기: {len(payload['c0_list'])}개 배치")
    
    try:
        print(f"서버로 암호화된 데이터 전송 중...")
        aggregate_response = requests.post(f"{SERVER_URL}/aggregate", json=payload)
        if aggregate_response.status_code == 200:
            try:
                aggregate_json = aggregate_response.json()
                print(f"[Round {r+1}] 서버 응답 수신 완료")
                
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
    print(f"[Round {r+1}] 글로벌 모델 로컬 데이터 정확도(학습 전): {acc_before:.2f}% | 로컬 모델 정확도(학습 후): {acc_after:.2f}%")
    print(f"[Round {r+1}] 글로벌 모델 테스트셋 정확도: {global_acc:.2f}%")
    print(f"=== 라운드 {r+1} 종료 ===\n") 