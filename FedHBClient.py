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

# device 설정
device = torch.device('cpu')  # GPU 환경 문제로 CPU 강제 지정

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
            state_dict = torch.load("global_model.pth", map_location=device)
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
    update = {
        'state_dict': updated_model.state_dict(),
        'num_samples': num_samples
    }
    torch.save(update, "update.pth")
    update_size = os.path.getsize("update.pth")
    files = {'file': open('update.pth', 'rb')}
    r_post = requests.post(f"{SERVER_URL}/submit_update", files=files)
    print(f"[Round {r+1}] 서버 응답: {r_post.json()}")
    try:
        aggregate_response = requests.post(f"{SERVER_URL}/aggregate")
        if aggregate_response.status_code == 200:
            try:
                aggregate_json = aggregate_response.json()
                print(f"[Round {r+1}] 집계 응답: {aggregate_json}")
            except:
                print(f"[Round {r+1}] 집계 응답: {aggregate_response.text}")
        else:
            print(f"[Round {r+1}] 집계 요청 실패: {aggregate_response.status_code}")
    except Exception as e:
        print(f"[Round {r+1}] 집계 요청 중 에러: {e}")
    print(f"[Round {r+1}] 로컬 손실: {avg_loss:.4f}, 에포크: {epochs}, 샘플 수: {num_samples}, 통신량: {update_size/1024:.2f} KB")
    print(f"[Round {r+1}] 글로벌 모델 로컬 데이터 정확도(학습 전): {acc_before:.2f}% | 로컬 모델 정확도(학습 후): {acc_after:.2f}%")
    os.remove("update.pth")
    print(f"[Round {r+1}] 테스트셋 정확도: {evaluate_local_accuracy(updated_model, test_loader, device):.2f}%")
    print(f"=== 라운드 {r+1} 종료 ===\n") 