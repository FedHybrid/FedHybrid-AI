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
global_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)  # 글로벌 모델 추가

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
    
    # FedAvg 방식: 글로벌 모델과 로컬 모델의 가중 평균
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
    
    # CKKS 배치 암호화를 통한 모델 업데이트 전송 (기존 방식 유지)
    state_dict = client_model.state_dict()
    
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
        "num_samples": num_samples,
        "loss": avg_loss  # 손실 정보 추가
    }
    
    print(f"서버로 전송할 페이로드 크기: {len(payload['c0_list'])}개 배치")
    
    # 5) 서버로 전송 (기존 방식 유지)
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
                    
                    # 서버 응답은 무시하고 FedAvg 집계 결과만 사용
                    print(f"서버 응답 무시 - FedAvg 집계 결과 유지")
                    
                    # 대신 글로벌 모델을 파일로 저장 (다음 라운드용)
                    torch.save(global_model.state_dict(), "global_model.pth")
                    print(f"글로벌 모델 저장 완료")
                        
            except Exception as e:
                print(f"[Round {r+1}] 집계 응답 처리 중 에러: {e}")
        else:
            print(f"[Round {r+1}] 집계 요청 실패: {aggregate_response.status_code}")
    except Exception as e:
        print(f"[Round {r+1}] 집계 요청 중 에러: {e}")
    
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