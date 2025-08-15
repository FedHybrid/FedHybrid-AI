#!/usr/bin/env python3
"""
정확도 향상을 위한 고급 모델 아키텍처
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class AdvancedDiabetesDataset(Dataset):
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
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x, y

class ResidualBlock(nn.Module):
    """잔차 연결 블록"""
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(out_features)
        self.batch_norm2 = nn.BatchNorm1d(out_features)
        
        # 잔차 연결을 위한 선형 변환
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.batch_norm2(out)
        out = F.relu(out + residual)
        out = self.dropout(out)
        
        return out

class AttentionModule(nn.Module):
    """어텐션 모듈"""
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        return attended_features, attention_weights

class AdvancedEnhancerModel(nn.Module):
    """고급 향상 모델"""
    def __init__(self, input_dim, num_classes=2, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(AdvancedEnhancerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 입력 레이어
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_rate)
            )
        
        # 어텐션 모듈
        self.attention = AttentionModule(hidden_dims[-1])
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier 초기화
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 입력에서 NaN/Inf 처리
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 입력 레이어
        x = self.input_layer(x)
        
        # 잔차 블록들
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # 어텐션 적용
        attended_features, attention_weights = self.attention(x)
        
        # 분류
        output = self.classifier(attended_features)
        
        # 출력에서 NaN/Inf 처리
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for 클래스 불균형 문제 해결"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def advanced_client_update(client_model, global_model, data_loader, criterion, round_idx, device, class_weights=None):
    """고급 클라이언트 업데이트 함수"""
    if len(data_loader.dataset) == 0:
        return client_model, float('inf'), 0, 0, 0.0
    
    # Focal Loss 사용 (클래스 불균형 해결)
    if class_weights is not None:
        focal_criterion = FocalLoss(alpha=class_weights[1], gamma=2)
    else:
        focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    # 모델 초기화 (첫 라운드에만)
    if round_idx == 0:
        client_model.apply(client_model._init_weights)
    
    client_model.train()
    
    # 고급 옵티마이저 설정
    optimizer = optim.AdamW(client_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    total_loss = 0.0
    total_samples = 0
    epochs = 5  # 에포크 수 증가
    
    # Early stopping
    best_loss = float('inf')
    patience = 5
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
            loss = focal_criterion(output, y)
            
            # 손실 NaN/Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print("경고: 손실에서 NaN/Inf 감지, 배치 건너뜀", flush=True)
                continue
            
            # L2 정규화
            l2_reg = 0.0
            for param in client_model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            # 정규화 후 손실 재체크
            if torch.isnan(loss) or torch.isinf(loss):
                print("경고: L2 정규화 후 손실에서 NaN/Inf 감지, 배치 건너뜀", flush=True)
                continue
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            total_samples += x.size(0)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        total_loss += epoch_loss
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # 정확도 계산
    client_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = client_model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    
    return client_model, avg_loss, epochs, total_samples, accuracy

def load_advanced_diabetes_data(csv_path, test_size=0.2, random_state=42):
    """고급 데이터 로딩 함수"""
    print("=== 고급 데이터 로딩 ===")
    
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
    
    # NaN 값 처리
    print(f"NaN 값 처리 전 - X에서 NaN 개수: {np.isnan(X).sum()}")
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"NaN 값 처리 후 - X에서 NaN 개수: {np.isnan(X).sum()}")
    
    print(f"원본 특성 수: {X.shape[1]}")
    
    # 상수 특성 제거
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)  # 분산이 0.01 이상인 특성만 선택
    X_variance_filtered = variance_selector.fit_transform(X)
    non_constant_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]
    
    print(f"상수 특성 제거 후 특성 수: {X_variance_filtered.shape[1]}")
    print(f"제거된 상수 특성: {[col for col in numeric_cols if col not in non_constant_features]}")
    
    # 특성 선택 (상위 8개로 고정)
    from sklearn.feature_selection import SelectKBest, f_classif
    k_features = min(8, X_variance_filtered.shape[1])  # 8개로 고정
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X_variance_filtered, y)
    selected_features = [non_constant_features[i] for i in selector.get_support(indices=True)]
    
    print(f"선택된 특성: {selected_features}")
    print(f"선택된 특성 수: {X_selected.shape[1]}")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 클래스 가중치 계산
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"클래스 가중치: {class_weights_dict}")
    
    # 데이터셋 생성
    train_dataset = AdvancedDiabetesDataset(X_train_scaled, y_train)
    test_dataset = AdvancedDiabetesDataset(X_test_scaled, y_test, scaler)
    
    return train_dataset, test_dataset, X_selected.shape[1], class_weights_dict, selected_features

if __name__ == "__main__":
    # 테스트
    print("고급 모델 테스트")
    model = AdvancedEnhancerModel(input_dim=15, num_classes=2)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
