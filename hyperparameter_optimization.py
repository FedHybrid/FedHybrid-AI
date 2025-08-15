#!/usr/bin/env python3
"""
하이퍼파라미터 최적화를 위한 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import optuna
from optuna.samplers import TPESampler
import json
import os

class OptimizedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OptimizedModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, activation='relu'):
        super(OptimizedModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_data_for_optimization(csv_path):
    """최적화를 위한 데이터 로딩"""
    df = pd.read_csv(csv_path)
    
    # 기본 전처리
    drop_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=drop_cols)
    df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    
    # 숫자형 특성만 선택
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'readmitted']
    
    X = df[numeric_cols].values
    y = df['readmitted'].values
    
    # 특성 선택 (상위 12개)
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=min(12, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def objective(trial):
    """Optuna 최적화 목적 함수"""
    
    # 하이퍼파라미터 정의
    params = {
        'hidden_dims': trial.suggest_categorical('hidden_dims', [
            [64, 32],
            [128, 64],
            [128, 64, 32],
            [256, 128, 64],
            [64, 32, 16],
            [128, 64, 32, 16]
        ]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epochs': trial.suggest_int('epochs', 5, 15),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'none'])
    }
    
    # 데이터 로딩
    X_train, X_test, y_train, y_test = load_data_for_optimization('diabetic_data.csv')
    
    # 데이터셋 생성
    train_dataset = OptimizedDataset(X_train, y_train)
    test_dataset = OptimizedDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 모델 생성
    model = OptimizedModel(
        input_dim=X_train.shape[1],
        hidden_dims=params['hidden_dims'],
        dropout_rate=params['dropout_rate'],
        activation=params['activation']
    )
    
    # 옵티마이저 설정
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    else:  # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], momentum=0.9)
    
    # 스케줄러 설정
    if params['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
    elif params['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['epochs']//3, gamma=0.5)
    else:
        scheduler = None
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    device = torch.device('cpu')
    model.to(device)
    
    best_accuracy = 0.0
    
    for epoch in range(params['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        best_accuracy = max(best_accuracy, accuracy)
        
        # 중간 보고
        trial.report(accuracy, epoch)
        
        # 조기 종료
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_accuracy

def run_hyperparameter_optimization(n_trials=100):
    """하이퍼파라미터 최적화 실행"""
    print("=== 하이퍼파라미터 최적화 시작 ===")
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1시간 타임아웃
    
    # 결과 출력
    print("=== 최적화 결과 ===")
    print(f"최고 정확도: {study.best_value:.4f}")
    print(f"최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 결과 저장
    results = {
        'best_accuracy': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
    }
    
    with open('hyperparameter_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과가 'hyperparameter_results.json'에 저장되었습니다.")
    
    return study.best_params

def evaluate_optimized_model(best_params, csv_path='diabetic_data.csv'):
    """최적화된 모델 평가"""
    print("=== 최적화된 모델 평가 ===")
    
    # 데이터 로딩
    X_train, X_test, y_train, y_test = load_data_for_optimization(csv_path)
    
    # 데이터셋 생성
    train_dataset = OptimizedDataset(X_train, y_train)
    test_dataset = OptimizedDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 모델 생성
    model = OptimizedModel(
        input_dim=X_train.shape[1],
        hidden_dims=best_params['hidden_dims'],
        dropout_rate=best_params['dropout_rate'],
        activation=best_params['activation']
    )
    
    # 옵티마이저 설정
    if best_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'], momentum=0.9)
    
    # 스케줄러 설정
    if best_params['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_params['epochs'])
    elif best_params['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=best_params['epochs']//3, gamma=0.5)
    else:
        scheduler = None
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    device = torch.device('cpu')
    model.to(device)
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(best_params['epochs']):
        # 학습
        model.train()
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        if scheduler:
            scheduler.step()
        
        # 검증
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        train_accuracy = train_correct / train_total
        test_accuracy = test_correct / test_total
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{best_params['epochs']}: Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    # 최종 평가
    final_accuracy = accuracy_score(all_labels, all_predictions)
    final_precision = precision_score(all_labels, all_predictions, average='weighted')
    final_recall = recall_score(all_labels, all_predictions, average='weighted')
    final_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"\n=== 최종 평가 결과 ===")
    print(f"정확도: {final_accuracy:.4f}")
    print(f"정밀도: {final_precision:.4f}")
    print(f"재현율: {final_recall:.4f}")
    print(f"F1 점수: {final_f1:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), 'optimized_model.pth')
    print(f"최적화된 모델이 'optimized_model.pth'에 저장되었습니다.")
    
    return {
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

if __name__ == "__main__":
    # 하이퍼파라미터 최적화 실행
    best_params = run_hyperparameter_optimization(n_trials=50)
    
    # 최적화된 모델 평가
    results = evaluate_optimized_model(best_params)
    
    print(f"\n✅ 하이퍼파라미터 최적화 완료!")
    print(f"최종 정확도: {results['accuracy']:.4f}")
