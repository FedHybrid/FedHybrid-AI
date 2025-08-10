import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Diabetes 데이터셋 로딩 함수 및 Dataset 클래스
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('int64')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_diabetes_data(csv_path, test_size=0.2, random_state=42):
    print(f"데이터 로드 시작: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"원본 데이터 크기: {df.shape}", flush=True)
    
    # 기본 전처리
    drop_cols = ['encounter_id', 'patient_nbr']
    available_drop_cols = [col for col in drop_cols if col in df.columns]
    if available_drop_cols:
        df = df.drop(columns=available_drop_cols)
        print(f"제거된 컬럼: {available_drop_cols}", flush=True)
    
    # readmitted 컬럼 처리
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    
    # 숫자형 컬럼만 feature로 사용
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'readmitted' in numeric_cols:
        numeric_cols.remove('readmitted')
    if 'max_glu_serum' in numeric_cols:
        numeric_cols.remove('max_glu_serum')  # 문제가 되는 컬럼 제거
    
    print(f"사용할 특성 컬럼 ({len(numeric_cols)}개): {numeric_cols[:5]}...", flush=True)
    
    # 특성 데이터 추출
    X = df[numeric_cols].values
    y = df['readmitted'].values if 'readmitted' in df.columns else np.zeros(len(df))
    
    print(f"특성 데이터 형태: {X.shape}, 레이블 형태: {y.shape}", flush=True)
    
    # 데이터 품질 검사 및 정리
    print("데이터 품질 검사 중...", flush=True)
    
    # 1. NaN 값 처리
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"NaN 값 발견: {nan_mask.sum()}개", flush=True)
        # 각 컬럼의 중앙값으로 NaN 대체
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.isnan(col_data).any():
                median_val = np.nanmedian(col_data)
                X[nan_mask[:, col_idx], col_idx] = median_val
                print(f"컬럼 {col_idx} NaN 값을 {median_val}로 대체", flush=True)
    
    # 2. 무한대 값 처리
    inf_mask = np.isinf(X)
    if inf_mask.any():
        print(f"무한대 값 발견: {inf_mask.sum()}개", flush=True)
        # 무한대 값을 해당 컬럼의 최대/최소 유한값으로 대체
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.isinf(col_data).any():
                finite_values = col_data[np.isfinite(col_data)]
                if len(finite_values) > 0:
                    max_finite = np.max(finite_values)
                    min_finite = np.min(finite_values)
                    X[X[:, col_idx] == np.inf, col_idx] = max_finite
                    X[X[:, col_idx] == -np.inf, col_idx] = min_finite
                    print(f"컬럼 {col_idx} 무한대 값을 {min_finite}~{max_finite} 범위로 대체", flush=True)
    
    # 3. 극값 처리 (IQR 방법)
    print("극값 처리 중...", flush=True)
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        q25, q75 = np.percentile(col_data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 3 * iqr  # 3 IQR로 완화
        upper_bound = q75 + 3 * iqr
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        if outlier_mask.any():
            X[col_data < lower_bound, col_idx] = lower_bound
            X[col_data > upper_bound, col_idx] = upper_bound
            print(f"컬럼 {col_idx}: {outlier_mask.sum()}개 극값을 [{lower_bound:.2f}, {upper_bound:.2f}] 범위로 클리핑", flush=True)
    
    # 4. 정규화 (StandardScaler 대신 간단한 Min-Max 정규화)
    print("데이터 정규화 중...", flush=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 정규화 후에도 NaN/Inf 체크
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print("경고: 정규화 후에도 NaN/Inf 값이 존재합니다. Min-Max 스케일링으로 변경합니다.", flush=True)
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        X_scaled = minmax_scaler.fit_transform(X)
        
        # 여전히 문제가 있다면 수동으로 처리
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("수동 정규화 수행 중...", flush=True)
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                col_min, col_max = np.min(col_data), np.max(col_data)
                if col_max - col_min > 0:
                    X_scaled[:, col_idx] = (col_data - col_min) / (col_max - col_min)
                else:
                    X_scaled[:, col_idx] = 0.5  # 모든 값이 동일한 경우
    
    # 최종 데이터 품질 확인
    print(f"최종 데이터 통계:", flush=True)
    print(f"  X 범위: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]", flush=True)
    print(f"  X 평균: {np.mean(X_scaled):.4f}, 표준편차: {np.std(X_scaled):.4f}", flush=True)
    print(f"  NaN 개수: {np.isnan(X_scaled).sum()}", flush=True)
    print(f"  Inf 개수: {np.isinf(X_scaled).sum()}", flush=True)
    print(f"  레이블 분포: {np.bincount(y)}", flush=True)
    
    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    
    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    print(f"학습 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개", flush=True)
    return train_dataset, test_dataset

# EnhancerModel 정의 (서버/클라이언트 공통)
class EnhancerModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dims=[64, 32]):  # 더 작은 hidden_dims
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout 감소
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 2. FedET 클래스 (Ensemble + Transfer)
# ----------------------------
class FedET:
    def __init__(self, input_dim, num_classes=3, num_clients=3, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.device = device
        
        # 클라이언트 모델들 (기본 모델)
        self.client_models = []
        for _ in range(num_clients):
            model = MLPClassifier(input_dim, num_classes)
            self.client_models.append(model.to(device))
        
        # Enhancer 모델 (앙상블 + 전이학습)
        self.enhancer = EnhancerModel(input_dim, num_classes).to(device)
        
        # 앙상블 가중치
        self.ensemble_weights = torch.ones(num_clients).to(device) / num_clients
        
        # 전이학습 관련
        self.transfer_learning_rate = 0.001
        self.ensemble_learning_rate = 0.01
        
    def train_enhancer(self, train_loader, round_idx):
        """Enhancer 모델 훈련 (앙상블 + 전이학습)"""
        self.enhancer.train()
        optimizer = optim.Adam(self.enhancer.parameters(), lr=self.transfer_learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # 1. 클라이언트 모델들의 예측 (앙상블)
            client_predictions = []
            with torch.no_grad():
                for client_model in self.client_models:
                    client_model.eval()
                    pred = client_model(x)
                    client_predictions.append(pred)
            
            # 2. 가중 앙상블 예측
            ensemble_pred = torch.zeros_like(client_predictions[0])
            for i, pred in enumerate(client_predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
            
            # 3. Enhancer 모델 훈련 (전이학습)
            optimizer.zero_grad()
            
            # 직접 예측
            enhancer_pred = self.enhancer(x)
            direct_loss = criterion(enhancer_pred, y)
            
            # 앙상블에서 전이학습
            transfer_loss = F.mse_loss(enhancer_pred, ensemble_pred.detach())
            
            # Knowledge Distillation (앙상블에서)
            temperature = 3.0
            ensemble_probs = torch.softmax(ensemble_pred / temperature, dim=1)
            enhancer_log_probs = torch.log_softmax(enhancer_pred / temperature, dim=1)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(enhancer_log_probs, ensemble_probs)
            
            # 전체 손실
            alpha = 0.4  # 직접 학습 가중치
            beta = 0.3   # 전이학습 가중치
            gamma = 0.3  # Knowledge Distillation 가중치
            
            total_loss_batch = (alpha * direct_loss + 
                              beta * transfer_loss + 
                              gamma * kd_loss)
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item() * x.size(0)
            total_samples += x.size(0)
        
        return total_loss / total_samples
    
    def update_ensemble_weights(self, validation_loader):
        """앙상블 가중치 업데이트"""
        self.enhancer.eval()
        client_accuracies = []
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                correct, total = 0, 0
                
                for x, y in validation_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = client_model(x)
                    pred_class = pred.argmax(1)
                    correct += (pred_class == y).sum().item()
                    total += y.size(0)
                
                accuracy = correct / total
                client_accuracies.append(accuracy)
        
        # 정확도 기반 가중치 업데이트
        accuracies = torch.tensor(client_accuracies, device=self.device)
        self.ensemble_weights = F.softmax(accuracies * 10, dim=0)  # 온도 스케일링
        
        return client_accuracies
    
    def predict_with_enhancer(self, x):
        """Enhancer 모델을 사용한 예측"""
        self.enhancer.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.enhancer(x)
    
    def get_ensemble_prediction(self, x):
        """앙상블 예측"""
        predictions = []
        with torch.no_grad():
            x = x.to(self.device)
            for client_model in self.client_models:
                client_model.eval()
                pred = client_model(x)
                predictions.append(pred)
            
            # 가중 앙상블
            ensemble_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
            
            return ensemble_pred

# ----------------------------
# 3. FedET 훈련 함수
# ----------------------------
def train_fedet(fedet, train_loaders, test_loader, num_rounds=50, local_epochs=5):
    """FedET 완전 훈련"""
    print("=== FedET (Federated Ensemble Transfer) 훈련 시작 ===")
    
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
        
        # 1. 클라이언트별 로컬 훈련
        client_losses = []
        for client_idx in range(fedet.num_clients):
            if client_idx < len(train_loaders):
                client_model = fedet.client_models[client_idx]
                train_loader = train_loaders[client_idx]
                
                # 로컬 훈련
                client_model.train()
                optimizer = optim.Adam(client_model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                total_loss = 0.0
                for epoch in range(local_epochs):
                    for x, y in train_loader:
                        x, y = x.to(fedet.device), y.to(fedet.device)
                        optimizer.zero_grad()
                        out = client_model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * x.size(0)
                
                avg_loss = total_loss / (len(train_loader.dataset) * local_epochs)
                client_losses.append(avg_loss)
                print(f"Client {client_idx + 1} Loss: {avg_loss:.4f}")
        
        # 2. Enhancer 모델 훈련 (앙상블 + 전이학습)
        if round_idx > 0:  # 첫 라운드 이후부터 enhancer 훈련
            # 모든 클라이언트 데이터를 합쳐서 enhancer 훈련
            combined_data = []
            for loader in train_loaders:
                combined_data.extend(loader.dataset)
            
            combined_loader = DataLoader(combined_data, batch_size=32, shuffle=True)
            enhancer_loss = fedet.train_enhancer(combined_loader, round_idx)
            print(f"Enhancer Loss: {enhancer_loss:.4f}")
            
            # 앙상블 가중치 업데이트
            accuracies = fedet.update_ensemble_weights(test_loader)
            print(f"Client Accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
            print(f"Ensemble Weights: {fedet.ensemble_weights.cpu().numpy()}")
        
        # 3. 평가
        if round_idx % 10 == 0:
            evaluate_fedet(fedet, test_loader, round_idx)

def evaluate_fedet(fedet, test_loader, round_idx):
    """FedET 모델 평가"""
    fedet.enhancer.eval()
    
    # Enhancer 모델 평가
    correct_enhancer, total = 0, 0
    correct_ensemble, total_ensemble = 0, 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(fedet.device), y.to(fedet.device)
            
            # Enhancer 예측
            enhancer_pred = fedet.predict_with_enhancer(x)
            enhancer_class = enhancer_pred.argmax(1)
            correct_enhancer += (enhancer_class == y).sum().item()
            
            # 앙상블 예측
            ensemble_pred = fedet.get_ensemble_prediction(x)
            ensemble_class = ensemble_pred.argmax(1)
            correct_ensemble += (ensemble_class == y).sum().item()
            
            total += y.size(0)
            total_ensemble += y.size(0)
    
    enhancer_acc = correct_enhancer / total * 100
    ensemble_acc = correct_ensemble / total_ensemble * 100
    
    print(f"\n=== Round {round_idx} 평가 결과 ===")
    print(f"Enhancer 모델 정확도: {enhancer_acc:.2f}%")
    print(f"앙상블 모델 정확도: {ensemble_acc:.2f}%")

# ----------------------------
# 4. 기존 모델 정의 (SimpleCNN)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# 5. 모델 압축 기법
# ----------------------------
class ModelQuantizer:
    def __init__(self, bits=8):
        self.bits = bits
        self.scale_factor = 2**(bits - 1) - 1

    def quantize_model(self, state_dict):
        quantized = {}
        scales = {}
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                mn, mx = tensor.min(), tensor.max()
                scale = (mx - mn) / (2 * self.scale_factor)
                scales[name] = (mn, scale)
                q = torch.round((tensor - mn) / scale - self.scale_factor).to(torch.int8)
                quantized[name] = q
            else:
                quantized[name] = tensor
        return quantized, scales

    def dequantize_model(self, quantized, scales):
        dequantized = {}
        for name, q in quantized.items():
            if name in scales:
                mn, scale = scales[name]
                dequantized[name] = (q.float() + self.scale_factor) * scale + mn
            else:
                dequantized[name] = q
        return dequantized

def top_k_sparsification(state_dict, k_ratio=0.1):
    sparse, idxs = {}, {}
    for name, tensor in state_dict.items():
        flat = tensor.flatten()
        k = max(1, int(len(flat) * k_ratio))
        _, top_idx = torch.topk(flat.abs(), k)
        sparse[name] = flat[top_idx]
        idxs[name] = (top_idx, tensor.shape)
    return sparse, idxs

def reconstruct_from_sparse(sparse, idxs):
    full = {}
    for name, vals in sparse.items():
        idx, shape = idxs[name]
        recon = torch.zeros(np.prod(shape), dtype=vals.dtype)
        recon[idx] = vals
        full[name] = recon.view(shape)
    return full

# ----------------------------
# 6. ALT: 적응형 로컬 epoch
# ----------------------------
def calculate_representation_similarity(local_model, global_model, data_loader, device):
    local_model.eval(); global_model.eval()
    sims = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            lf = local_model.features(x).view(x.size(0), -1)
            gf = global_model.features(x).view(x.size(0), -1)
            sims.append(F.cosine_similarity(lf, gf, dim=1).mean().item())
    return np.mean(sims)

def adaptive_local_epochs(similarity, round_idx, total_rounds, base=5):
    sf = max(0.5, similarity)
    pf = 1 + (round_idx / total_rounds) * 0.5
    e = int(base * sf * pf)
    return max(1, min(10, e))

def alt_client_update(client_model, global_model, data_loader, criterion,
                      round_idx, total_rounds, device):
    sim = calculate_representation_similarity(client_model, global_model, data_loader, device)
    epochs = adaptive_local_epochs(sim, round_idx, total_rounds)
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    total_loss = 0.0
    for _ in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
    return client_model, total_loss / (len(data_loader.dataset) * epochs), epochs, sim

def client_update_full(client_model, global_model, data_loader, criterion, round_idx, device, use_kd=True, use_fedprox=True, use_pruning=False):
    if len(data_loader.dataset) == 0:
        return client_model, float('inf'), 0, 0
    
    # 간소화된 클래스 가중치 계산 (속도 향상)
    num_classes = 2
    class_counts = torch.zeros(num_classes)
    for _, y in data_loader:
        for i in range(num_classes):
            class_counts[i] += (y == i).sum()
    
    # 간단한 가중치 계산
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"  클래스 분포: {class_counts}")
    print(f"  클래스 가중치: {class_weights}")
    
    # 가중 손실 함수 사용
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 모델 가중치 초기화는 첫 라운드에만 수행
    if round_idx == 0:
        for param in client_model.parameters():
            if len(param.shape) > 1:  # 가중치 행렬
                torch.nn.init.xavier_uniform_(param)
            else:  # 바이어스
                torch.nn.init.zeros_(param)
    
    client_model.train()
    optimizer = optim.Adam(client_model.parameters(), lr=0.001, weight_decay=1e-5)  # 학습률 감소 (안정성 향상)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # 더 보수적인 스케줄링
    total_loss = 0.0
    total_samples = 0
    mu = 0.001  # FedProx 파라미터 더 감소 (안정성 향상)
    epochs = 5  # 에포크 수 복원 (안정성 향상)
    
    # Early stopping 변수
    best_loss = float('inf')
    patience = 2  # Early stopping 조건 완화 (속도 향상)
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = client_model(x)
            loss = weighted_criterion(output, y)  # 가중 손실 사용
            
            # FedProx (활성화)
            if use_fedprox and round_idx > 0:
                prox_loss = 0.0
                for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                    prox_loss += ((w - w_t.detach()) ** 2).sum()
                loss += mu * prox_loss
            
            # Knowledge Distillation (활성화)
            if use_kd and round_idx > 0:
                with torch.no_grad():
                    global_model.eval()
                    temperature = 3.0 * np.exp(-0.1 * round_idx)
                    teacher_probs = torch.softmax(global_model(x) / temperature, dim=1)
                student_log_probs = torch.log_softmax(output / temperature, dim=1)
                kd_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)
                loss = 0.8 * loss + 0.2 * kd_loss  # KD 가중치 감소 (안정성 향상)
            
            loss.backward()
            
            # NaN/Inf 체크 (더 상세히)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  경고: NaN/Inf 손실 감지 (loss: {loss.item()}), 배치 건너뛰기", flush=True)
                optimizer.zero_grad()  # 그래디언트 초기화
                continue
            
            # 그래디언트 NaN/Inf 체크
            has_nan_grad = False
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  경고: {name}에서 NaN/Inf 그래디언트 감지", flush=True)
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"  경고: NaN/Inf 그래디언트 감지, 배치 건너뛰기", flush=True)
                optimizer.zero_grad()
                continue
            
            # 그래디언트 클리핑 강화
            grad_norm = torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=0.1)
            if grad_norm > 10.0:  # 그래디언트 노름이 너무 큰 경우
                print(f"  경고: 큰 그래디언트 노름 감지 ({grad_norm:.2f}), 배치 건너뛰기", flush=True)
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            # 모델 파라미터 NaN/Inf 체크 (업데이트 후)
            has_nan_params = False
            for name, param in client_model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  경고: {name}에서 NaN/Inf 파라미터 감지", flush=True)
                    has_nan_params = True
                    break
            
            if has_nan_params:
                print(f"  경고: 모델 파라미터에 NaN/Inf 감지, 학습 중단", flush=True)
                break
            
            epoch_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        # 학습률 스케줄링은 에포크 끝에서 수행
        
        # Early stopping 체크
        if total_samples > 0:
            avg_epoch_loss = epoch_loss / total_samples
            if np.isnan(avg_epoch_loss) or np.isinf(avg_epoch_loss):
                print(f"  경고: NaN/Inf 손실 감지, 학습 중단")
                break
        else:
            print(f"  경고: 유효한 샘플이 없음, 학습 중단")
            break
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (loss: {avg_epoch_loss:.4f})")
            break
        
        # 학습률 스케줄링 (에포크 끝에서)
        scheduler.step()
        
        # 에포크별 손실 출력 (디버깅용)
        if epoch % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.6f}")
        
        total_loss += epoch_loss
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
    else:
        avg_loss = float('inf')  # 기본값 설정
    return client_model, avg_loss, epochs, total_samples

def download_global_model():
    for _ in range(5):  # 최대 5번 재시도
        r = requests.get(f"{SERVER_URL}/get_model")
        with open("global_model.pth", "wb") as f:
            f.write(r.content)
        if os.path.getsize("global_model.pth") > 1000:  # 1KB 이상이면 정상
            break
        print("global_model.pth 파일이 너무 작음, 재시도...")
        time.sleep(1)
    state_dict = torch.load("global_model.pth", map_location=device)
    client_model.load_state_dict(state_dict)
    os.remove("global_model.pth")

# MLPClassifier, load_cancer_data 등은 더 이상 사용하지 않으므로 주석 처리 또는 삭제
# 서버와 클라이언트가 모두 EnhancerModel, load_diabetes_data만 사용하도록 유지
# (필요시 load_diabetes_data 함수는 FedHBClient.py에서 model.py로 옮겨도 됨)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 단순한 2층 네트워크로 변경
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 드롭아웃 감소
            prev_dim = h
        
        # 출력층
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 간단한 초기화
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 7. FedET 실행 예제
# ----------------------------
if __name__ == "__main__":
    # 데이터 로드
    train_dataset, test_dataset = load_diabetes_data('diabetes_patient_data.csv')
    input_dim = train_dataset.X.shape[1]
    
    # 클라이언트별 데이터 분할
    num_clients = 3
    client_datasets = []
    samples_per_client = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_dataset)
        client_data = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # FedET 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fedet = FedET(input_dim=input_dim, num_classes=3, num_clients=num_clients, device=device)
    
    # FedET 훈련
    train_fedet(fedet, train_loaders, test_loader, num_rounds=30, local_epochs=3)
    
    # 최종 평가
    print("\n=== 최종 FedET 모델 평가 ===")
    evaluate_fedet(fedet, test_loader, 30) 