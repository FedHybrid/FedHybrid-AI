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

# CKKS 파라미터 설정
Delta = 2**6  # 스케일 팩터
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

# 데이터셋 준비 (train/test 분할)
train_dataset, test_dataset = load_diabetes_data('diabetic_data.csv')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
input_dim = train_dataset.X.shape[1]

# 모델 준비 (EnhancerModel)
client_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)

SERVER_URL = "http://localhost:8000"
NUM_ROUNDS = 200

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
    updated_model, avg_loss, epochs, num_samples = client_update_full(
        client_model, client_model, train_loader, nn.CrossEntropyLoss(), r, device,
        use_kd=False, use_fedprox=False, use_pruning=False
    )
    acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
    
    # CKKS 배치 암호화를 통한 모델 업데이트 전송
    state_dict = updated_model.state_dict()
    
    print(f"\n=== 클라이언트: 모델 암호화 시작 ===")
    print(f"모델 파라미터 수: {len(state_dict)}개 레이어")
    
    # 1) Tensor → flat numpy vector
    flat = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])
    print(f"평면화된 벡터 크기: {len(flat)}")
    print(f"원본 값 범위: {flat.min():.4f} ~ {flat.max():.4f}")
    print(f"원본 값 평균: {flat.mean():.4f}")
    
    # 2) plaintext 다항식 계수로 인코딩
    m_coeffs = flat.astype(np.complex128)  # 복소 슬롯 벡터
    
    # 3) 배치 암호화
    c0_list, c1_list = batch_encrypt(m_coeffs, batch_size=4)
    print(f"암호화 완료: {len(c0_list)}개 배치")
    
    # 4) 리스트로 변환 후 JSON 직렬화
    payload = {
        "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list],
        "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list],
        "original_size": len(m_coeffs),
        "num_samples": num_samples
    }
    
    print(f"서버로 전송할 페이로드 크기: {len(payload['c0_list'])}개 배치")
    
    # 5) 서버로 전송
    try:
        print(f"서버로 업데이트 전송 중...")
        aggregate_response = requests.post(f"{SERVER_URL}/aggregate", json=payload)
        if aggregate_response.status_code == 200:
            try:
                aggregate_json = aggregate_response.json()
                print(f"[Round {r+1}] 서버 응답 수신")
                
                # 서버로부터 암호화된 집계 모델 수신 및 복호화
                if "c0_list" in aggregate_json and "c1_list" in aggregate_json:
                    print(f"=== 클라이언트: 서버 응답 복호화 시작 ===")
                    c0_list_agg = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in aggregate_json["c0_list"]]
                    c1_list_agg = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in aggregate_json["c1_list"]]
                    original_size = aggregate_json["original_size"]
                    
                    print(f"받은 암호문: {len(c0_list_agg)}개 배치")
                    
                    # 배치 복호화
                    m_vals = batch_decrypt(c0_list_agg, c1_list_agg, original_size, batch_size=4)
                    print(f"복호화 완료: {len(m_vals)}개 값")
                    print(f"복호화된 값 범위: {m_vals.real.min():.4f} ~ {m_vals.real.max():.4f}")
                    print(f"복호화된 값 평균: {m_vals.real.mean():.4f}")
                    
                    # 파라미터 복원
                    ptr = 0
                    for k, v in client_model.state_dict().items():
                        numel = v.numel()
                        arr = torch.from_numpy(
                            m_vals[ptr:ptr+numel].astype(np.float32)
                        ).view(v.size())
                        client_model.state_dict()[k].copy_(arr.to(device))
                        ptr += numel
                        print(f"파라미터 {k} 복원: shape={v.size()}, 값 범위={arr.min().item():.4f}~{arr.max().item():.4f}")
                    
                    print(f"모델 파라미터 복원 완료")
                        
            except Exception as e:
                print(f"[Round {r+1}] 집계 응답 처리 중 에러: {e}")
        else:
            print(f"[Round {r+1}] 집계 요청 실패: {aggregate_response.status_code}")
    except Exception as e:
        print(f"[Round {r+1}] 집계 요청 중 에러: {e}")
    
    print(f"[Round {r+1}] 로컬 손실: {avg_loss:.4f}, 에포크: {epochs}, 샘플 수: {num_samples}")
    print(f"[Round {r+1}] 글로벌 모델 로컬 데이터 정확도(학습 전): {acc_before:.2f}% | 로컬 모델 정확도(학습 후): {acc_after:.2f}%")
    print(f"[Round {r+1}] 테스트셋 정확도: {evaluate_local_accuracy(client_model, test_loader, device):.2f}%")
    print(f"=== 라운드 {r+1} 종료 ===\n") 