#!/usr/bin/env python3
"""
정확도 향상을 위한 개선된 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

class ImprovedDiabetesDataset(Dataset):
    def __init__(self, X, y, scaler=None):
        if scaler is not None:
            self.X = scaler.transform(X).astype('float32')
        else:
            self.X = X.astype('float32')
        
        # NaN/Inf 값 처리
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.y = y.astype('int64')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        
        # 텐서에서도 NaN/Inf 처리
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x, y

def load_improved_diabetes_data(csv_path, test_size=0.2, random_state=42):
    """개선된 데이터 로딩 함수"""
    print("=== 개선된 데이터 로딩 ===")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    drop_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=drop_cols)
    df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'readmitted']
    
    X = df[numeric_cols].values
    y = df['readmitted'].values
    
    print(f"원본 특성 수: {X.shape[1]}")
    
    # 1. 고정된 특성 선택 (일관성 보장)
    # 가장 중요한 특성들을 미리 정의하여 차원 일치 보장
    fixed_features = [
        'admission_source_id', 'time_in_hospital', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    
    # 사용 가능한 특성만 선택
    available_features = [col for col in fixed_features if col in numeric_cols]
    
    # 부족한 경우 다른 특성 추가
    if len(available_features) < 8:
        remaining_cols = [col for col in numeric_cols if col not in available_features]
        available_features.extend(remaining_cols[:8-len(available_features)])
    
    # 정확히 8개 특성만 사용
    selected_features = available_features[:8]
    
    # 선택된 특성의 인덱스 찾기
    feature_indices = [numeric_cols.index(col) for col in selected_features]
    X_selected = X[:, feature_indices]
    
    print(f"고정 선택된 특성: {selected_features}")
    print(f"선택된 특성 수: {X_selected.shape[1]} (고정)")
    print(f"특성 인덱스: {feature_indices}")
    
    # 2. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. 스케일링 전 NaN/Inf 체크
    print(f"스케일링 전 NaN 체크:")
    print(f"  X_train NaN 개수: {np.isnan(X_train).sum()}")
    print(f"  X_test NaN 개수: {np.isnan(X_test).sum()}")
    print(f"  X_train Inf 개수: {np.isinf(X_train).sum()}")
    print(f"  X_test Inf 개수: {np.isinf(X_test).sum()}")
    
    # NaN/Inf 값을 0으로 대체
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. 스케일링 (RobustScaler 사용 - 이상치에 강함)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 스케일링 후에도 NaN/Inf 체크 및 처리
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    
    print(f"스케일링 완료")
    print(f"스케일링 후 NaN 체크:")
    print(f"  X_train_scaled NaN 개수: {np.isnan(X_train_scaled).sum()}")
    print(f"  X_test_scaled NaN 개수: {np.isnan(X_test_scaled).sum()}")
    print(f"훈련 세트: {X_train.shape[0]:,}개 샘플")
    print(f"테스트 세트: {X_test.shape[0]:,}개 샘플")
    
    # 4. 클래스 분포 확인
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    print(f"훈련 세트 클래스 분포:")
    for cls, count in zip(train_unique, train_counts):
        percentage = count / len(y_train) * 100
        print(f"  클래스 {cls}: {count:,}개 ({percentage:.1f}%)")
    
    # 5. 클래스 가중치 계산
    class_weights = compute_class_weights(y_train)
    print(f"클래스 가중치: {class_weights}")
    
    train_dataset = ImprovedDiabetesDataset(X_train_scaled, y_train)
    test_dataset = ImprovedDiabetesDataset(X_test_scaled, y_test)
    
    return train_dataset, test_dataset, class_weights, selected_features

def compute_class_weights(y):
    """클래스 가중치 계산"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    return weights

class ImprovedEnhancerModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        
        # 더 간단하고 안정적인 네트워크
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.classifier = nn.Linear(32, num_classes)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He 초기화 (ReLU에 더 적합)
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 입력에서 NaN/Inf 처리 (조용히 처리)
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        features = self.feature_extractor(x)
        
        # 중간 출력 NaN/Inf 처리 (조용히 처리)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        output = self.classifier(features)
        
        # 최종 출력 NaN/Inf 처리 (조용히 처리)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output

def improved_client_update(client_model, global_model, data_loader, criterion, round_idx, device, class_weights=None):
    """개선된 클라이언트 업데이트 함수"""
    if len(data_loader.dataset) == 0:
        return client_model, float('inf'), 0, 0
    
    # 클래스 가중치가 있으면 사용
    if class_weights is not None:
        weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
        weighted_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        weighted_criterion = criterion
    
    # 모델 초기화 (첫 라운드에만)
    if round_idx == 0:
        client_model.apply(client_model._init_weights)
    
    client_model.train()
    
    # 개선된 옵티마이저 설정
    optimizer = optim.AdamW(client_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    
    total_loss = 0.0
    total_samples = 0
    epochs = 5  # 에포크 수 증가
    
    # Early stopping
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # 입력 데이터 NaN/Inf 체크
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            optimizer.zero_grad()
            
            output = client_model(x)
            loss = weighted_criterion(output, y)
            
            # 손실 NaN/Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print("경고: 손실에서 NaN/Inf 감지, 배치 건너뜀", flush=True)
                continue
            
            # L2 정규화 (FedProx 대신)
            l2_reg = 0.0
            for param in client_model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            # 정규화 후 손실 재체크
            if torch.isnan(loss) or torch.isinf(loss):
                print("경고: L2 정규화 후 손실에서 NaN/Inf 감지, 배치 건너뜀", flush=True)
                continue
            
            loss.backward()
            
            # 그래디언트 NaN/Inf 체크
            has_nan_grad = False
            for param in client_model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("경고: 그래디언트에서 NaN/Inf 감지, 배치 건너뜀", flush=True)
                optimizer.zero_grad()
                continue
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        scheduler.step()
        
        # Early stopping 체크
        if total_samples > 0:
            avg_epoch_loss = epoch_loss / total_samples
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (loss: {avg_epoch_loss:.4f})")
                break
    
    # 학습 완료 후 정확도 계산
    client_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # 입력 데이터 NaN/Inf 체크
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            outputs = client_model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = best_loss if best_loss != float('inf') else epoch_loss / total_samples
    
    print(f"라운드 {round_idx} 정확도: {accuracy:.4f} ({correct}/{total})", flush=True)
    
    return client_model, avg_loss, epoch + 1, total_samples, accuracy 