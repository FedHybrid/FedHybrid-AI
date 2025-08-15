#!/usr/bin/env python3
"""
최적화된 하이퍼파라미터를 사용하는 개선된 FedHybrid 클라이언트
"""

import torch
import requests
from advanced_model import AdvancedEnhancerModel, advanced_client_update, load_advanced_diabetes_data
from aggregation import CommunicationEfficientFedHB
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import time
from ckks import batch_encrypt, batch_decrypt
import argparse
import sys
import io
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image

# device 설정
device = torch.device('cpu')

# 클라이언트 설정
CLIENT_ID = os.getenv('CLIENT_ID', 'client_1')

# CKKS 파라미터 설정
z_q = 1 << 10
rescale_q = z_q
N = 4
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)

SERVER_URL = "http://localhost:8000"
NUM_ROUNDS = 5

# 최적화된 하이퍼파라미터
OPTIMIZED_PARAMS = {
    'hidden_dims': [256, 128, 64],
    'dropout_rate': 0.174,
    'learning_rate': 0.0027,
    'weight_decay': 7.19e-5,
    'batch_size': 64,
    'epochs': 13,
    'activation': 'relu',
    'optimizer': 'adamw',
    'scheduler': 'cosine'
}

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
    for attempt in range(5):
        try:
            print(f"글로벌 모델 다운로드 시도 {attempt + 1}/5...")
            r = requests.get(f"{SERVER_URL}/get_model", timeout=10)
            
            if r.status_code == 200:
                with open("global_model.pth", "wb") as f:
                    f.write(r.content)
                
                file_size = os.path.getsize("global_model.pth")
                print(f"글로벌 모델 다운로드 완료 (파일 크기: {file_size} bytes)")
                
                try:
                    model_data = torch.load("global_model.pth", map_location=device, weights_only=False)
                    
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        print(f"모델 메타데이터: {model_data.get('model_type', 'Unknown')} v{model_data.get('version', 'Unknown')}")
                        print(f"서버 모델 입력 차원: {model_data.get('input_dim', 'Unknown')}")
                        state_dict = model_data['state_dict']
                    else:
                        print("구 형식의 모델 파일입니다.")
                        state_dict = model_data
                    
                    os.remove("global_model.pth")
                    print("글로벌 모델 로드 성공")
                    return state_dict
                except Exception as e:
                    print(f"글로벌 모델 파일 로드 실패: {e}")
                    if os.path.exists("global_model.pth"):
                        os.remove("global_model.pth")
            else:
                print(f"서버 응답 오류: {r.status_code} - {r.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"서버 연결 실패 (시도 {attempt + 1}/5)")
        except requests.exceptions.Timeout:
            print(f"서버 응답 시간 초과 (시도 {attempt + 1}/5)")
        except Exception as e:
            print(f"글로벌 모델 다운로드 중 오류: {e}")
        
        if attempt < 4:
            print("3초 후 재시도...")
            time.sleep(3)
    
    raise RuntimeError("글로벌 모델을 정상적으로 다운로드하지 못했습니다.")

def analyze_feature_importance(model, data_loader, feature_names, device):
    """특성 중요도 분석"""
    model.eval()
    feature_importance = {}
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x.requires_grad_(True)
            
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            
            # 그래디언트의 절댓값 평균으로 중요도 계산
            gradients = x.grad.abs().mean(dim=0)
            
            for i, feature_name in enumerate(feature_names):
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = []
                feature_importance[feature_name].append(gradients[i].item())
            
            break  # 첫 번째 배치만 사용
    
    # 평균 중요도 계산
    avg_importance = {}
    for feature_name, values in feature_importance.items():
        avg_importance[feature_name] = np.mean(values)
    
    # 중요도 순으로 정렬
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance

def predict_diabetes_probability_with_explanation(model, data_loader, feature_names, device):
    """당뇨병 확률 예측 및 설명"""
    model.eval()
    probabilities = []
    predictions = []
    feature_importance = analyze_feature_importance(model, data_loader, feature_names, device)
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            
            # 당뇨병 확률 (클래스 1)
            diabetes_probs = probs[:, 1].cpu().numpy()
            probabilities.extend(diabetes_probs)
            
            # 예측 클래스
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(probabilities), np.array(predictions), feature_importance

def save_results_to_excel_with_graphs(original_data, probabilities, predictions, feature_importance=None, output_path='prediction_results.xlsx', round_accuracies=None, round_losses=None):
    """결과를 엑셀 파일로 저장 (그래프 포함)"""
    try:
        print(f"결과 저장 시작: {len(probabilities)}개 데이터")
        
        # NaN 값 처리
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        predictions = np.nan_to_num(predictions, nan=0, posinf=1, neginf=0).astype(int)
        
        # 데이터 크기 제한 (메모리 및 시간 절약)
        max_rows = 10000  # 최대 10,000행으로 제한
        if len(original_data) > max_rows:
            print(f"데이터 크기가 큽니다. 상위 {max_rows}개 행만 저장합니다.")
            # 확률 기준으로 상위 데이터만 선택
            top_indices = np.argsort(probabilities)[-max_rows:]
            original_data = original_data.iloc[top_indices]
            probabilities = probabilities[top_indices]
            predictions = predictions[top_indices]
        
        # 원본 데이터에 예측 결과 추가
        result_df = original_data.copy()
        
        # 불필요한 Unnamed 컬럼들 제거
        unnamed_cols = [col for col in result_df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            print(f"불필요한 컬럼 제거: {unnamed_cols}")
            result_df = result_df.drop(columns=unnamed_cols)
        
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        print(f"엑셀 파일 저장 시작: {len(result_df)}행")
        
        # 엑셀 파일 생성 (그래프 포함)
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 데이터 시트
                result_df.to_excel(writer, sheet_name='예측_결과', index=False)
                
                # 통계 시트
                stats_data = {
                    '통계': [
                        '총 데이터 수',
                        '당뇨병 예측 수',
                        '정상 예측 수',
                        '평균 당뇨병 확률',
                        '최대 당뇨병 확률',
                        '최소 당뇨병 확률'
                    ],
                    '값': [
                        len(result_df),
                        sum(predictions),
                        len(predictions) - sum(predictions),
                        f"{np.mean(probabilities):.4f}",
                        f"{np.max(probabilities):.4f}",
                        f"{np.min(probabilities):.4f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='통계', index=False)
                
                # 특성 중요도 시트
                if feature_importance:
                    importance_data = {
                        '특성명': [feature for feature, _ in feature_importance],
                        '중요도': [importance for _, importance in feature_importance]
                    }
                    importance_df = pd.DataFrame(importance_data)
                    importance_df.to_excel(writer, sheet_name='특성_중요도', index=False)
                
                # 라운드별 정확도 시트
                if round_accuracies and round_losses:
                    round_data = {
                        '라운드': list(range(1, len(round_accuracies) + 1)),
                        '정확도(%)': round_accuracies,
                        '손실': round_losses
                    }
                    round_df = pd.DataFrame(round_data)
                    round_df.to_excel(writer, sheet_name='라운드별_성능', index=False)
                
                # 워크북 가져오기
                workbook = writer.book
                
                # 확률 분포 히스토그램
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title('당뇨병 확률 분포', fontsize=14, fontweight='bold')
                    plt.xlabel('당뇨병 확률', fontsize=12)
                    plt.ylabel('빈도', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # 이미지를 바이트로 저장
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    # 엑셀에 이미지 추가
                    worksheet = writer.sheets['예측_결과']
                    img = openpyxl.drawing.image.Image(img_buffer)
                    img.anchor = f'A{len(result_df) + 3}'
                    worksheet.add_image(img)
                    
                    plt.close()
                    print("확률 분포 히스토그램 추가 완료")
                except Exception as e:
                    print(f"히스토그램 생성 실패: {e}")
                
                # 특성 중요도 막대 그래프
                if feature_importance:
                    try:
                        plt.figure(figsize=(12, 8))
                        features = [feature for feature, _ in feature_importance[:10]]  # 상위 10개
                        importances = [importance for _, importance in feature_importance[:10]]
                        
                        plt.barh(range(len(features)), importances, color='lightcoral')
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('중요도', fontsize=12)
                        plt.title('특성 중요도 (상위 10개)', fontsize=14, fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        
                        # 이미지를 바이트로 저장
                        img_buffer2 = io.BytesIO()
                        plt.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
                        img_buffer2.seek(0)
                        
                        # 엑셀에 이미지 추가
                        worksheet = writer.sheets['특성_중요도']
                        img2 = openpyxl.drawing.image.Image(img_buffer2)
                        img2.anchor = 'A1'
                        worksheet.add_image(img2)
                        
                        plt.close()
                        print("특성 중요도 그래프 추가 완료")
                    except Exception as e:
                        print(f"특성 중요도 그래프 생성 실패: {e}")
                
                # 라운드별 정확도 및 손실 차트
                if round_accuracies and round_losses:
                    try:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # 정확도 차트
                        rounds = list(range(1, len(round_accuracies) + 1))
                        ax1.plot(rounds, round_accuracies, 'o-', color='blue', linewidth=2, markersize=6)
                        ax1.set_title('라운드별 정확도 변화', fontsize=14, fontweight='bold')
                        ax1.set_xlabel('라운드', fontsize=12)
                        ax1.set_ylabel('정확도 (%)', fontsize=12)
                        ax1.grid(True, alpha=0.3)
                        ax1.set_ylim(0, 100)
                        
                        # 각 점에 값 표시
                        for i, acc in enumerate(round_accuracies):
                            ax1.annotate(f'{acc:.1f}%', (rounds[i], acc), 
                                       textcoords="offset points", xytext=(0,10), 
                                       ha='center', fontsize=10)
                        
                        # 손실 차트
                        ax2.plot(rounds, round_losses, 'o-', color='red', linewidth=2, markersize=6)
                        ax2.set_title('라운드별 손실 변화', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('라운드', fontsize=12)
                        ax2.set_ylabel('손실', fontsize=12)
                        ax2.grid(True, alpha=0.3)
                        
                        # 각 점에 값 표시
                        for i, loss in enumerate(round_losses):
                            ax2.annotate(f'{loss:.4f}', (rounds[i], loss), 
                                       textcoords="offset points", xytext=(0,10), 
                                       ha='center', fontsize=10)
                        
                        plt.tight_layout()
                        
                        # 이미지를 바이트로 저장
                        img_buffer3 = io.BytesIO()
                        plt.savefig(img_buffer3, format='png', dpi=300, bbox_inches='tight')
                        img_buffer3.seek(0)
                        
                        # 엑셀에 이미지 추가
                        worksheet = writer.sheets['라운드별_성능']
                        img3 = openpyxl.drawing.image.Image(img_buffer3)
                        img3.anchor = 'A1'
                        worksheet.add_image(img3)
                        
                        plt.close()
                        print("라운드별 성능 차트 추가 완료")
                    except Exception as e:
                        print(f"라운드별 성능 차트 생성 실패: {e}")
            
            print(f"엑셀 파일 저장 완료: {output_path}")
            print(f"파일 크기: {os.path.getsize(output_path)} bytes")
            return True
            
        except Exception as excel_error:
            print(f"엑셀 저장 실패, CSV로 대체 저장: {excel_error}")
            csv_path = output_path.replace('.xlsx', '.csv')
            result_df.to_csv(csv_path, index=False)
            print(f"CSV 파일 저장 완료: {csv_path}")
            return False
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='FedHybrid 클라이언트 (최적화 버전)')
    parser.add_argument('--input_file', type=str, required=True, help='입력 CSV 파일 경로')
    args = parser.parse_args()
    
    data_file = args.input_file
    
    if not os.path.exists(data_file):
        print(f"❌ 파일을 찾을 수 없습니다: {data_file}")
        return False
    
    print(f"=== FedHybrid 클라이언트 (최적화 버전) ===")
    print(f"📁 입력 파일: {data_file}")
    print(f"🔧 최적화된 하이퍼파라미터 사용")
    print(f"  - Hidden dims: {OPTIMIZED_PARAMS['hidden_dims']}")
    print(f"  - Dropout rate: {OPTIMIZED_PARAMS['dropout_rate']}")
    print(f"  - Learning rate: {OPTIMIZED_PARAMS['learning_rate']}")
    print(f"  - Batch size: {OPTIMIZED_PARAMS['batch_size']}")
    print(f"  - Epochs: {OPTIMIZED_PARAMS['epochs']}")
    
    # 데이터 로딩
    try:
        train_dataset, test_dataset, input_dim, class_weights, selected_features = load_advanced_diabetes_data(data_file)
        train_loader = DataLoader(train_dataset, batch_size=OPTIMIZED_PARAMS['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=OPTIMIZED_PARAMS['batch_size'], shuffle=False, num_workers=0)
        print(f"✅ 고급 데이터 로드 완료 - 입력 차원: {input_dim}")
        print(f"선택된 특성: {selected_features}")
        print(f"클래스 가중치: {class_weights}")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return False

    # 모델 준비 (최적화된 파라미터 사용)
    client_model = AdvancedEnhancerModel(
        input_dim=input_dim, 
        num_classes=2,
        hidden_dims=OPTIMIZED_PARAMS['hidden_dims'],
        dropout_rate=OPTIMIZED_PARAMS['dropout_rate']
    ).to(device)
    
    global_model = AdvancedEnhancerModel(
        input_dim=input_dim, 
        num_classes=2,
        hidden_dims=OPTIMIZED_PARAMS['hidden_dims'],
        dropout_rate=OPTIMIZED_PARAMS['dropout_rate']
    ).to(device)

    print(f"=== {NUM_ROUNDS}라운드 학습 시작 ===")
    
    # 라운드별 정확도 추적
    round_accuracies = []
    round_losses = []
    
    for r in range(NUM_ROUNDS):
        round_start_time = time.time()
        print(f"\n🚀 === 라운드 {r+1}/{NUM_ROUNDS} 시작 ===")
        print(f"⏰ 시작 시간: {time.strftime('%H:%M:%S')}")
        
        # 1단계: 글로벌 모델 다운로드
        print(f"📥 1단계: 서버에서 글로벌 모델 다운로드 중...")
        try:
            state_dict = download_global_model()
            
            try:
                global_model.load_state_dict(state_dict)
                print(f"✅ 글로벌 모델 다운로드 및 로드 성공")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"❌ 모델 차원 불일치: {e}")
                    print("🔄 로컬 모델 초기화로 진행합니다.")
                    global_model.load_state_dict(client_model.state_dict())
                else:
                    raise e
        except Exception as e:
            print(f"❌ 글로벌 모델 다운로드/로드 실패: {e}")
            print("🔄 로컬 모델 초기화로 진행합니다.")
            global_model.load_state_dict(client_model.state_dict())
        
        acc_before = evaluate_local_accuracy(client_model, train_loader, device)
        
        # 2단계: 로컬 학습 수행 (최적화된 파라미터 사용)
        print(f"🎓 2단계: 로컬 모델 학습 시작...")
        training_start_time = time.time()
        
        try:
            updated_model, avg_loss, epochs, num_samples, accuracy = advanced_client_update(
                client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device, class_weights
            )
            print(f"✅ 고급 학습 함수 사용 완료")
        except Exception as e:
            print(f"❌ 고급 학습 실패: {e}")
            return False
            
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
        
        # 학습된 모델을 클라이언트 모델에 복사
        client_model.load_state_dict(updated_model.state_dict())
        
        # 3단계: CKKS 암호화
        encryption_start_time = time.time()
        print(f"\n🔐 3단계: 클라이언트 데이터 CKKS 암호화")
        state_dict = client_model.state_dict()
        print(f"📦 모델 파라미터 수: {len(state_dict)}개 레이어")
        
        # Tensor → flat numpy vector
        flat_params = []
        for param_name, param_tensor in state_dict.items():
            flat_params.extend(param_tensor.cpu().numpy().flatten())
        
        flat_params = np.array(flat_params, dtype=np.float32)
        print(f"📊 평면화된 파라미터 크기: {flat_params.shape}")
        
        # CKKS 암호화
        encrypted_params = batch_encrypt(flat_params, batch_size=4)
        print(f"🔒 암호화된 파라미터 배치 수: {len(encrypted_params[0])}")
        
        encryption_end_time = time.time()
        encryption_duration = encryption_end_time - encryption_start_time
        
        # 4단계: 서버로 전송
        upload_start_time = time.time()
        print(f"\n📤 4단계: 암호화된 파라미터 서버 전송")
        
        try:
            # 암호화된 파라미터를 JSON으로 직렬화 (서버 UpdateRequest 모델에 맞춤)
            encrypted_data = {
                'client_id': CLIENT_ID,
                'round_id': r + 1,
                'c0_list': [c0.tolist() for c0 in encrypted_params[0]],
                'c1_list': [c1.tolist() for c1 in encrypted_params[1]],
                'original_size': len(flat_params),
                'num_samples': num_samples,
                'loss': float(avg_loss),
                'accuracy': float(accuracy)
            }
            
            response = requests.post(f"{SERVER_URL}/upload", json=encrypted_data, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ 서버 전송 성공")
                server_response = response.json()
                print(f"📊 서버 응답: {server_response}")
            else:
                print(f"❌ 서버 전송 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 서버 전송 중 오류: {e}")
            return False
        
        upload_end_time = time.time()
        upload_duration = upload_end_time - upload_start_time
        
        # 라운드 완료 요약
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        
        print(f"\n=== 라운드 {r+1} 완료 ===")
        print(f"⏱️  총 소요 시간: {round_duration:.2f}초")
        print(f"  - 학습 시간: {training_duration:.2f}초")
        print(f"  - 암호화 시간: {encryption_duration:.2f}초")
        print(f"  - 전송 시간: {upload_duration:.2f}초")
        print(f"📈 정확도 변화: {acc_before:.2f}% → {acc_after:.2f}%")
        print(f"🎯 최종 정확도: {accuracy:.2f}%")
        print(f"📊 평균 손실: {avg_loss:.4f}")
        print(f"📚 학습 에포크: {epochs}")
        print(f"📊 샘플 수: {num_samples}")
        
        # 라운드별 정확도 및 손실 저장
        round_accuracies.append(accuracy)
        round_losses.append(avg_loss)
        
        # 실시간 정확도 차트 출력 (라운드 5개마다)
        if (r + 1) % 5 == 0 or r == NUM_ROUNDS - 1:
            print(f"\n📈 === 라운드 {r+1}까지의 정확도 추이 ===")
            for i, acc in enumerate(round_accuracies):
                print(f"  라운드 {i+1}: {acc:.2f}%")
            print(f"  평균 정확도: {np.mean(round_accuracies):.2f}%")
            print(f"  최고 정확도: {np.max(round_accuracies):.2f}%")
    
    # 최종 모델 평가
    print(f"\n=== 최종 모델 평가 ===")
    final_accuracy = evaluate_local_accuracy(client_model, test_loader, device)
    print(f"🎯 최종 테스트 정확도: {final_accuracy:.2f}%")
    
    # 예측 수행
    print(f"\n=== 예측 수행 ===")
    try:
        # 원본 데이터 로드
        df = pd.read_csv(data_file)
        drop_cols = ['encounter_id', 'patient_nbr']
        df = df.drop(columns=drop_cols)
        df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
        
        # 숫자형 특성만 선택
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'readmitted']
        
        X = df[numeric_cols].values
        y = df['readmitted'].values
        
        # 상수 특성 제거 후 특성 선택 (8개로 고정)
        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
        
        # 상수 특성 제거
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = variance_selector.fit_transform(X)
        non_constant_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]
        
        # 특성 선택 (8개로 고정)
        k_features = min(8, X_variance_filtered.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_selected = selector.fit_transform(X_variance_filtered, y)
        selected_features_for_prediction = [non_constant_features[i] for i in selector.get_support(indices=True)]
        
        # 스케일링
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 예측용 데이터셋 생성
        from advanced_model import AdvancedDiabetesDataset
        prediction_dataset = AdvancedDiabetesDataset(X_scaled, np.zeros(len(X_scaled)))
        prediction_loader = DataLoader(prediction_dataset, batch_size=OPTIMIZED_PARAMS['batch_size'], shuffle=False)
        
        # 예측 수행
        probabilities, predictions, feature_importance = predict_diabetes_probability_with_explanation(
            client_model, prediction_loader, selected_features_for_prediction, device
        )
        
        # 결과 분석
        diabetes_count = np.sum(predictions == 1)
        normal_count = np.sum(predictions == 0)
        avg_probability = np.mean(probabilities)
        
        print(f"📊 예측 결과:")
        print(f"  - 총 데이터: {len(predictions):,}개")
        print(f"  - 당뇨병 예측: {diabetes_count:,}개 ({diabetes_count/len(predictions)*100:.1f}%)")
        print(f"  - 정상 예측: {normal_count:,}개 ({normal_count/len(predictions)*100:.1f}%)")
        print(f"  - 평균 당뇨병 확률: {avg_probability:.1%}")
        
        # 특성 중요도 출력
        print(f"\n🔍 특성 중요도 (상위 10개):")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        # 결과를 엑셀 파일로 저장 (그래프 포함)
        success = save_results_to_excel_with_graphs(
            df, probabilities, predictions, feature_importance, 
            'prediction_results_optimized.xlsx', round_accuracies, round_losses
        )
        
        if success:
            print(f"\n💾 예측 결과가 'prediction_results_optimized.xlsx'에 저장되었습니다.")
        else:
            print(f"\n💾 예측 결과가 'prediction_results_optimized.csv'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 예측 수행 중 오류: {e}")
    
    print(f"\n🎉 FedHybrid 클라이언트 (최적화 버전) 완료!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
