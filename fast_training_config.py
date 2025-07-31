#!/usr/bin/env python3
"""
빠른 학습을 위한 설정 파일
"""

# 학습 속도 최적화 설정
FAST_TRAINING_CONFIG = {
    # 배치 크기 설정
    "batch_size": 128,  # 더 큰 배치 크기
    
    # 에포크 설정
    "epochs": 2,  # 더 적은 에포크
    
    # 학습률 설정
    "learning_rate": 0.005,  # 더 높은 학습률
    
    # Early stopping 설정
    "patience": 1,  # 더 빠른 early stopping
    
    # 모델 구조 설정
    "hidden_dims": [32, 16],  # 더 작은 모델
    
    # Dropout 설정
    "dropout_rate": 0.1,  # 더 낮은 dropout
    
    # 최적화 설정
    "use_mixed_precision": True,  # 혼합 정밀도 사용
    "use_gradient_accumulation": False,  # 그래디언트 누적 비활성화
    
    # 데이터 로더 설정
    "num_workers": 0,  # CPU 환경에서는 0
    "pin_memory": False,  # CPU 환경에서는 False
}

# 극한 속도 설정 (정확도는 다소 떨어질 수 있음)
ULTRA_FAST_CONFIG = {
    "batch_size": 256,
    "epochs": 1,
    "learning_rate": 0.01,
    "patience": 1,
    "hidden_dims": [16],
    "dropout_rate": 0.0,
    "use_mixed_precision": True,
    "use_gradient_accumulation": False,
    "num_workers": 0,
    "pin_memory": False,
}

def get_training_config(mode="fast"):
    """학습 설정 반환"""
    if mode == "ultra_fast":
        return ULTRA_FAST_CONFIG
    else:
        return FAST_TRAINING_CONFIG 