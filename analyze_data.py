#!/usr/bin/env python3
"""
데이터셋 분석 스크립트
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def analyze_diabetes_data(csv_path='diabetic_data.csv'):
    """당뇨병 데이터셋 분석"""
    print("=== 당뇨병 데이터셋 분석 ===")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"전체 데이터 크기: {df.shape}")
    
    # 기본 전처리
    drop_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=drop_cols)
    df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
    
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'readmitted']
    
    X = df[numeric_cols].values
    y = df['readmitted'].values
    
    print(f"특성 수: {X.shape[1]}")
    print(f"샘플 수: {X.shape[0]}")
    
    # 클래스 분포 분석
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n클래스 분포:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        percentage = count / len(y) * 100
        print(f"  클래스 {cls}: {count:,}개 ({percentage:.1f}%)")
    
    # 데이터 품질 분석
    print(f"\n데이터 품질:")
    print(f"  결측값: {np.isnan(X).sum()}")
    print(f"  무한값: {np.isinf(X).sum()}")
    
    # 특성별 분석
    print(f"\n특성별 통계:")
    for i, col in enumerate(numeric_cols):
        values = X[:, i]
        print(f"  {col}:")
        print(f"    평균: {np.mean(values):.3f}")
        print(f"    표준편차: {np.std(values):.3f}")
        print(f"    최소값: {np.min(values):.3f}")
        print(f"    최대값: {np.max(values):.3f}")
        print(f"    고유값 수: {len(np.unique(values))}")
    
    # 상관관계 분석
    print(f"\n상관관계 분석:")
    correlation_matrix = np.corrcoef(X.T)
    target_correlations = correlation_matrix[-1, :-1]  # 마지막 행 (타겟)과의 상관관계
    
    # 상관관계가 높은 상위 10개 특성
    top_features_idx = np.argsort(np.abs(target_correlations))[-10:][::-1]
    print("타겟과 상관관계가 높은 상위 10개 특성:")
    for i, idx in enumerate(top_features_idx):
        corr = target_correlations[idx]
        print(f"  {i+1}. {numeric_cols[idx]}: {corr:.3f}")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n훈련/테스트 분할:")
    print(f"  훈련 세트: {X_train.shape[0]:,}개 샘플")
    print(f"  테스트 세트: {X_test.shape[0]:,}개 샘플")
    
    # 훈련 세트 클래스 분포
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    print(f"  훈련 세트 클래스 분포:")
    for cls, count in zip(train_unique, train_counts):
        percentage = count / len(y_train) * 100
        print(f"    클래스 {cls}: {count:,}개 ({percentage:.1f}%)")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': numeric_cols,
        'correlations': target_correlations
    }

if __name__ == "__main__":
    data_info = analyze_diabetes_data() 