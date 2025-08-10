#!/usr/bin/env python3
"""
개선된 모델 테스트 스크립트
"""

import torch
import numpy as np
from improved_model import ImprovedEnhancerModel, load_improved_diabetes_data
from torch.utils.data import DataLoader

def test_model():
    """모델 기본 동작 테스트"""
    print("=== 모델 테스트 시작 ===")
    
    # 디바이스 설정
    device = torch.device('cpu')
    print(f"디바이스: {device}")
    
    # 간단한 테스트 데이터 생성
    input_dim = 8  # 개선된 모델의 특성 수
    batch_size = 5
    num_classes = 2
    
    # 모델 생성
    model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)
    print(f"모델 생성 완료: {sum(p.numel() for p in model.parameters())}개 파라미터")
    
    # 테스트 입력 생성 (정규화된 데이터)
    test_input = torch.randn(batch_size, input_dim).to(device) * 0.5  # 작은 값으로 제한
    print(f"테스트 입력 형태: {test_input.shape}")
    print(f"입력 범위: [{test_input.min().item():.4f}, {test_input.max().item():.4f}]")
    
    # 모델 실행
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        probs = torch.softmax(output, dim=1)
        predictions = torch.argmax(output, dim=1)
    
    print(f"출력 형태: {output.shape}")
    print(f"출력 범위: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"확률 합계: {probs.sum(dim=1)}")
    print(f"예측: {predictions}")
    print(f"확률: {probs}")
    
    # NaN 체크
    if torch.isnan(output).any():
        print("❌ 출력에 NaN 감지!")
        return False
    else:
        print("✅ 출력에 NaN 없음")
    
    if torch.isinf(output).any():
        print("❌ 출력에 Inf 감지!")
        return False
    else:
        print("✅ 출력에 Inf 없음")
    
    print("=== 모델 테스트 성공 ===")
    return True

def test_data_loading():
    """데이터 로딩 테스트"""
    print("\n=== 데이터 로딩 테스트 시작 ===")
    
    try:
        # 테스트 데이터 파일이 있는지 확인
        import os
        test_files = ['diabetic_data.csv', 'diabetic.csv', 'test-data.csv']
        data_file = None
        
        for file in test_files:
            if os.path.exists(file):
                data_file = file
                break
        
        if data_file is None:
            print("테스트 데이터 파일을 찾을 수 없습니다.")
            return False
        
        print(f"데이터 파일 사용: {data_file}")
        
        # 데이터 로딩
        train_dataset, test_dataset, class_weights, selected_features = load_improved_diabetes_data(data_file)
        
        print(f"학습 데이터: {len(train_dataset)}개")
        print(f"테스트 데이터: {len(test_dataset)}개")
        print(f"선택된 특성: {selected_features}")
        print(f"클래스 가중치: {class_weights}")
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 첫 번째 배치 테스트
        for x, y in train_loader:
            print(f"배치 형태: X={x.shape}, y={y.shape}")
            print(f"X 범위: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"y 값: {y[:5]}")
            
            # NaN 체크
            if torch.isnan(x).any():
                print("❌ 입력 데이터에 NaN 감지!")
                return False
            else:
                print("✅ 입력 데이터에 NaN 없음")
            
            break  # 첫 번째 배치만 테스트
        
        print("=== 데이터 로딩 테스트 성공 ===")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 로딩 테스트 실패: {e}")
        return False

def test_training_step():
    """학습 단계 테스트"""
    print("\n=== 학습 단계 테스트 시작 ===")
    
    try:
        # 모델 생성
        device = torch.device('cpu')
        model = ImprovedEnhancerModel(input_dim=8, num_classes=2).to(device)
        
        # 더미 데이터 생성
        batch_size = 10
        x = torch.randn(batch_size, 8) * 0.5  # 작은 값으로 제한
        y = torch.randint(0, 2, (batch_size,))
        
        # 옵티마이저 및 손실 함수
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 학습 단계
        model.train()
        optimizer.zero_grad()
        
        output = model(x)
        loss = criterion(output, y)
        
        print(f"손실: {loss.item():.4f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ 손실에 NaN/Inf 감지!")
            return False
        
        loss.backward()
        
        # 그래디언트 체크
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print("❌ 그래디언트에 NaN/Inf 감지!")
                    return False
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"그래디언트 노름: {total_grad_norm:.4f}")
        
        optimizer.step()
        
        print("✅ 학습 단계 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 학습 단계 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🧪 FedHybrid-AI 모델 테스트")
    print("=" * 50)
    
    success = True
    
    # 1. 모델 기본 동작 테스트
    if not test_model():
        success = False
    
    # 2. 데이터 로딩 테스트
    if not test_data_loading():
        success = False
    
    # 3. 학습 단계 테스트
    if not test_training_step():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
    else:
        print("❌ 일부 테스트 실패. 문제를 해결해주세요.")
    
    print("=" * 50)
