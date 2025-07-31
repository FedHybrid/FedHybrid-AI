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
        self.y = y.astype('int64')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

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
    
    # 1. 특성 선택 (상위 8개 특성만 사용)
    selector = SelectKBest(score_func=f_classif, k=8)
    X_selected = selector.fit_transform(X, y)
    selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
    
    print(f"선택된 특성: {selected_features}")
    print(f"선택된 특성 수: {X_selected.shape[1]}")
    
    # 2. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. 스케일링 (RobustScaler 사용 - 이상치에 강함)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"스케일링 완료")
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
        
        # 더 깊고 넓은 네트워크
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
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
            optimizer.zero_grad()
            
            output = client_model(x)
            loss = weighted_criterion(output, y)
            
            # L2 정규화 (FedProx 대신)
            l2_reg = 0.0
            for param in client_model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            loss.backward()
            
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
    
    avg_loss = best_loss if best_loss != float('inf') else epoch_loss / total_samples
    return client_model, avg_loss, epoch + 1, total_samples 