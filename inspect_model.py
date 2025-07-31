#!/usr/bin/env python3
"""
암호화된 모델 파일 분석 스크립트
"""

import torch
import numpy as np
import os
import sys

def analyze_encrypted_model(file_path):
    """암호화된 모델 파일 분석"""
    print(f"=== {file_path} 파일 분석 ===")
    
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return
    
    try:
        # 파일 로드
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        print(f"📁 파일 타입: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔑 키 목록: {list(data.keys())}")
            
            # original_size 확인
            if 'original_size' in data:
                print(f"📏 원본 크기: {data['original_size']:,} 파라미터")
            
            # c0_list, c1_list 확인
            if 'c0_list' in data and 'c1_list' in data:
                c0_list = data['c0_list']
                c1_list = data['c1_list']
                
                print(f"🔐 c0_list 길이: {len(c0_list)} 배치")
                print(f"🔐 c1_list 길이: {len(c1_list)} 배치")
                
                if len(c0_list) > 0:
                    print(f"📊 첫 번째 배치 c0 형태: {c0_list[0].shape}")
                    print(f"📊 첫 번째 배치 c1 형태: {c1_list[0].shape}")
                    print(f"📊 첫 번째 배치 c0 값: {c0_list[0]}")
                    print(f"📊 첫 번째 배치 c1 값: {c1_list[0]}")
                    
                    # 값 범위 확인
                    c0_values = np.concatenate([c0.flatten() for c0 in c0_list])
                    c1_values = np.concatenate([c1.flatten() for c1 in c1_list])
                    
                    print(f"📈 c0 전체 값 범위: {c0_values.real.min():.3f} ~ {c0_values.real.max():.3f}")
                    print(f"📈 c1 전체 값 범위: {c1_values.real.min():.3f} ~ {c1_values.real.max():.3f}")
                    print(f"📈 c0 평균: {c0_values.real.mean():.3f}")
                    print(f"📈 c1 평균: {c1_values.real.mean():.3f}")
        
        elif isinstance(data, torch.nn.Module):
            print("🧠 일반 PyTorch 모델입니다")
            print(f"📊 모델 파라미터 수: {sum(p.numel() for p in data.parameters()):,}")
            
        else:
            print(f"❓ 알 수 없는 데이터 타입: {type(data)}")
            
    except Exception as e:
        print(f"❌ 파일 로드 중 오류 발생: {e}")

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "encrypted_global_model.pth"
    
    analyze_encrypted_model(file_path)
    
    # 다른 모델 파일들도 확인
    model_files = [
        "global_model.pth",
        "encrypted_global_model.pth"
    ]
    
    print("\n" + "="*50)
    print("📂 현재 디렉토리의 모델 파일들:")
    
    for file in model_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"📄 {file} ({file_size:.2f} MB)")
        else:
            print(f"❌ {file} (존재하지 않음)")

if __name__ == "__main__":
    main() 