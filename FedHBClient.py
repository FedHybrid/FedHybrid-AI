import torch
import requests
from model import client_update_full, load_diabetes_data
from improved_model import ImprovedEnhancerModel
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

# device 설정
device = torch.device('cpu')  # GPU 환경 문제로 CPU 강제 지정

# 클라이언트 설정
import os
CLIENT_ID = os.getenv('CLIENT_ID', 'client_1')  # 환경변수로 클라이언트 ID 설정 가능

# CKKS 파라미터 설정 (ckks.py와 동일하게)
z_q = 1 << 10   # 2^10 = 1,024 (평문 인코딩용 스케일)
rescale_q = z_q  # 리스케일링용 스케일
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

SERVER_URL = "http://localhost:8000"
NUM_ROUNDS = 5

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
                
                # 파일 크기 확인
                file_size = os.path.getsize("global_model.pth")
                print(f"글로벌 모델 다운로드 완료 (파일 크기: {file_size} bytes)")
                
                try:
                    model_data = torch.load("global_model.pth", map_location=device, weights_only=False)
                    
                    # 새 형식 (메타데이터 포함)인지 확인
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        print(f"모델 메타데이터: {model_data.get('model_type', 'Unknown')} v{model_data.get('version', 'Unknown')}")
                        print(f"서버 모델 입력 차원: {model_data.get('input_dim', 'Unknown')}")
                        state_dict = model_data['state_dict']
                    else:
                        # 구 형식 (state_dict만)
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
        
        if attempt < 4:  # 마지막 시도가 아니면 대기
            print("3초 후 재시도...")
            time.sleep(3)
    
    raise RuntimeError("글로벌 모델을 정상적으로 다운로드하지 못했습니다. 서버가 실행 중인지 확인해주세요.")

def analyze_feature_importance(model, data_loader, feature_names, device):
    """특성 중요도 분석"""
    model.eval()
    feature_importance = {}
    
    print("=== 특성 중요도 분석 시작 ===")
    
    with torch.no_grad():
        # 첫 번째 배치로 특성 중요도 계산
        for x, _ in data_loader:
            x = x.to(device)
            x.requires_grad_(True)
            
            # 원본 예측
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            original_prob = probs[:, 1].mean()  # 당뇨병 확률
            
            # 각 특성별 중요도 계산
            for i, feature_name in enumerate(feature_names):
                # 특성값을 0으로 설정
                x_modified = x.clone()
                x_modified[:, i] = 0
                
                # 수정된 예측
                outputs_modified = model(x_modified)
                probs_modified = torch.softmax(outputs_modified, dim=1)
                modified_prob = probs_modified[:, 1].mean()
                
                # 중요도 = 원본 확률 - 수정된 확률
                importance = abs(original_prob - modified_prob).item()
                feature_importance[feature_name] = importance
            
            break  # 첫 번째 배치만 사용
    
    # 중요도 순으로 정렬
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("특성 중요도 (높은 순):")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")
    
    return feature_importance

def explain_prediction(model, sample_data, feature_names, device):
    """개별 예측에 대한 설명"""
    model.eval()
    
    with torch.no_grad():
        x = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        diabetes_prob = probs[0, 1].item()
        
        print(f"\n=== 개별 예측 설명 ===")
        print(f"당뇨병 확률: {diabetes_prob:.4f}")
        
        # 각 특성의 기여도 계산
        contributions = {}
        for i, feature_name in enumerate(feature_names):
            x_modified = x.clone()
            x_modified[0, i] = 0  # 특성값을 0으로 설정
            
            outputs_modified = model(x_modified)
            probs_modified = torch.softmax(outputs_modified, dim=1)
            modified_prob = probs_modified[0, 1].item()
            
            contribution = diabetes_prob - modified_prob
            contributions[feature_name] = contribution
        
        # 기여도 순으로 정렬
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("특성별 기여도:")
        for feature, contribution in sorted_contributions[:5]:  # 상위 5개만
            direction = "증가" if contribution > 0 else "감소"
            print(f"  {feature}: {contribution:.4f} ({direction})")
        
        return contributions

def explain_prediction_process(model, sample_data, feature_names, device):
    """예측 과정을 단계별로 설명"""
    model.eval()
    
    print(f"\n=== 예측 과정 상세 설명 ===")
    
    with torch.no_grad():
        x = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 1단계: 입력 데이터 확인
        print(f"1단계: 입력 데이터")
        print(f"  입력 형태: {x.shape}")
        print(f"  특성 개수: {len(feature_names)}")
        print(f"  입력값 범위: {x.min().item():.2f} ~ {x.max().item():.2f}")
        
        # 2단계: 모델 통과
        print(f"\n2단계: 모델 통과")
        outputs = model(x)
        print(f"  모델 출력 (로짓): {outputs}")
        print(f"  출력 형태: {outputs.shape}")
        
        # 3단계: 소프트맥스 적용
        probs = torch.softmax(outputs, dim=1)
        print(f"\n3단계: 소프트맥스 적용")
        print(f"  소프트맥스 결과: {probs}")
        print(f"  확률 합계: {probs.sum().item():.4f}")
        
        # 4단계: 최종 예측
        diabetes_prob = probs[0, 1].item()
        normal_prob = probs[0, 0].item()
        predicted_class = torch.argmax(outputs, dim=1).item()
        
        print(f"\n4단계: 최종 예측")
        print(f"  정상 확률: {normal_prob:.4f}")
        print(f"  당뇨병 확률: {diabetes_prob:.4f}")
        print(f"  예측 클래스: {predicted_class} ({'당뇨병' if predicted_class == 1 else '정상'})")
        
        # 5단계: 특성별 기여도 분석
        print(f"\n5단계: 특성별 기여도 분석")
        contributions = {}
        for i, feature_name in enumerate(feature_names):
            x_modified = x.clone()
            x_modified[0, i] = 0  # 특성값을 0으로 설정
            
            outputs_modified = model(x_modified)
            probs_modified = torch.softmax(outputs_modified, dim=1)
            modified_prob = probs_modified[0, 1].item()
            
            contribution = diabetes_prob - modified_prob
            contributions[feature_name] = contribution
            
            print(f"  {feature_name}: {contribution:+.4f} (원본: {diabetes_prob:.4f} → 수정: {modified_prob:.4f})")
        
        # 6단계: 해석
        print(f"\n6단계: 예측 해석")
        if diabetes_prob > 0.5:
            print(f"  → 당뇨병 위험이 높음 (확률: {diabetes_prob:.1%})")
        else:
            print(f"  → 정상 범위 (당뇨병 확률: {diabetes_prob:.1%})")
        
        # 가장 중요한 특성들
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"  주요 영향 특성:")
        for i, (feature, contribution) in enumerate(sorted_contributions[:3]):
            direction = "증가" if contribution > 0 else "감소"
            print(f"    {i+1}. {feature}: {contribution:+.4f} ({direction})")
        
        return {
            'diabetes_prob': diabetes_prob,
            'predicted_class': predicted_class,
            'contributions': contributions
        }

def predict_diabetes_probability_with_explanation(model, data_loader, feature_names, device):
    """해석 가능한 당뇨병 확률 예측"""
    model.eval()
    probabilities = []
    predictions = []
    explanations = []
    
    print(f"=== 해석 가능한 예측 시작 ===")
    print(f"모델 상태: {model.training}")
    print(f"디바이스: {device}")
    print(f"특성 개수: {len(feature_names)}")
    
    # 특성 중요도 분석
    feature_importance = analyze_feature_importance(model, data_loader, feature_names, device)
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            outputs = model(x)
            
            # 디버깅: 첫 번째 배치의 출력 확인
            if batch_idx == 0:
                print(f"\n첫 번째 배치 분석:")
                print(f"  입력 형태: {x.shape}")
                print(f"  모델 출력 형태: {outputs.shape}")
                print(f"  출력 샘플: {outputs[:3]}")
            
            probs = torch.softmax(outputs, dim=1)
            
            # 디버깅: 첫 번째 배치의 확률 확인
            if batch_idx == 0:
                print(f"  확률 형태: {probs.shape}")
                print(f"  확률 샘플: {probs[:3]}")
                print(f"  확률 합계: {probs.sum(dim=1)[:3]}")
            
            batch_probs = probs[:, 1].cpu().numpy()  # 당뇨병 확률 (클래스 1)
            _, predicted = torch.max(outputs, 1)
            batch_preds = predicted.cpu().numpy()
            
            probabilities.extend(batch_probs)
            predictions.extend(batch_preds)
            
            # 첫 번째 배치의 첫 번째 샘플에 대한 상세 설명
            if batch_idx == 0:
                print(f"\n첫 번째 샘플 상세 분석:")
                sample_data = x[0].cpu().numpy()
                sample_explanation = explain_prediction_process(model, sample_data, feature_names, device)
                explanations.append(sample_explanation)
            
            # 디버깅: 첫 번째 배치의 예측 확인
            if batch_idx == 0:
                print(f"  예측: {batch_preds[:3]}")
                print(f"  당뇨병 확률: {batch_probs[:3]}")
    
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    
    print(f"\n=== 전체 예측 완료 ===")
    print(f"확률 범위: {probabilities.min():.4f} ~ {probabilities.max():.4f}")
    print(f"확률 평균: {probabilities.mean():.4f}")
    print(f"예측 분포: {np.bincount(predictions)}")
    print(f"고유 확률 값 개수: {len(np.unique(probabilities))}")
    
    return probabilities, predictions, feature_importance

def save_results_to_excel(original_data, probabilities, predictions, feature_importance=None, output_path='prediction_results.xlsx'):
    """결과를 엑셀 파일로 저장 (간소화 버전)"""
    try:
        print(f"결과 저장 시작: {len(probabilities)}개 데이터", flush=True)
        
        # NaN 값 처리
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        predictions = np.nan_to_num(predictions, nan=0, posinf=1, neginf=0).astype(int)
        
        # 데이터 크기 제한 (메모리 및 시간 절약)
        max_rows = 10000  # 최대 10,000행으로 제한
        if len(original_data) > max_rows:
            print(f"데이터 크기가 큽니다. 상위 {max_rows}개 행만 저장합니다.", flush=True)
            # 확률 기준으로 상위 데이터만 선택
            top_indices = np.argsort(probabilities)[-max_rows:]
            original_data = original_data.iloc[top_indices]
            probabilities = probabilities[top_indices]
            predictions = predictions[top_indices]
        
        # 원본 데이터에 예측 결과 추가
        result_df = original_data.copy()
        
        # 불필요한 Unnamed 컬럼들 제거 (Unnamed:50, Unnamed:51, Unnamed:52 등)
        unnamed_cols = [col for col in result_df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            print(f"불필요한 컬럼 제거: {unnamed_cols}", flush=True)
            result_df = result_df.drop(columns=unnamed_cols)
        
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        print(f"엑셀 파일 저장 시작: {len(result_df)}행", flush=True)
        
        # 간단한 엑셀 저장 (시트 하나만)
        try:
            result_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"엑셀 파일 저장 완료: {output_path}", flush=True)
        except Exception as excel_error:
            print(f"엑셀 저장 실패, CSV로 대체 저장: {excel_error}", flush=True)
            csv_path = output_path.replace('.xlsx', '.csv')
            result_df.to_csv(csv_path, index=False)
            print(f"CSV 파일 저장 완료: {csv_path}", flush=True)
            return True
        
        # 기본 통계 출력
        print(f"결과가 {output_path}에 저장되었습니다.", flush=True)
        print(f"총 {len(result_df)}개 데이터에 대한 예측 완료", flush=True)
        print(f"당뇨병 예측: {sum(predictions)}개", flush=True)
        print(f"정상 예측: {len(predictions) - sum(predictions)}개", flush=True)
        print(f"평균 당뇨병 확률: {np.mean(probabilities):.4f}", flush=True)
        
        if os.path.exists(output_path):
            print(f"파일 크기: {os.path.getsize(output_path)} bytes", flush=True)
        
        return True
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(input_file=None):
    """메인 실행 함수"""
    print("=== FedHybrid 클라이언트 시작 ===", flush=True)
    
    # 입력 파일 처리
    if input_file and os.path.exists(input_file):
        print(f"입력 파일: {input_file}", flush=True)
        data_file = input_file
    else:
        print("기본 데이터 파일 사용: diabetic_data.csv", flush=True)
        data_file = 'diabetic_data.csv'
    
    # 데이터셋 준비 (개선된 버전 사용)
    try:
        from improved_model import load_improved_diabetes_data
        train_dataset, test_dataset, class_weights, selected_features = load_improved_diabetes_data(data_file)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        input_dim = train_dataset.X.shape[1]
        print(f"개선된 데이터 로드 완료 - 입력 차원: {input_dim}", flush=True)
        print(f"선택된 특성: {selected_features}", flush=True)
        print(f"클래스 가중치: {class_weights}", flush=True)
    except Exception as e:
        print(f"개선된 데이터 로드 실패, 기본 버전 사용: {e}", flush=True)
        try:
            train_dataset, test_dataset = load_diabetes_data(data_file)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            input_dim = train_dataset.X.shape[1]
            class_weights = None
            selected_features = None
            print(f"기본 데이터 로드 완료 - 입력 차원: {input_dim}", flush=True)
        except Exception as e2:
            print(f"데이터 로드 완전 실패: {e2}", flush=True)
            return False

    # 모델 준비 (EnhancerModel)
    client_model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=2).to(device)
    global_model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=2).to(device)  # 글로벌 모델 추가

    print(f"=== {NUM_ROUNDS}라운드 학습 시작 ===", flush=True)
    
    for r in range(NUM_ROUNDS):
        round_start_time = time.time()  # 라운드 시작 시간
        print(f"\n🚀 === 라운드 {r+1}/{NUM_ROUNDS} 시작 ===", flush=True)
        print(f"⏰ 시작 시간: {time.strftime('%H:%M:%S')}", flush=True)
        
        # 1단계: 글로벌 모델 다운로드
        print(f"📥 1단계: 서버에서 글로벌 모델 다운로드 중...", flush=True)
        try:
            state_dict = download_global_model()
            
            # 서버 모델과 클라이언트 모델의 차원이 다른 경우 처리
            try:
                global_model.load_state_dict(state_dict)
                print(f"✅ 글로벌 모델 다운로드 및 로드 성공", flush=True)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"❌ 모델 차원 불일치: {e}", flush=True)
                    print("🔄 로컬 모델 초기화로 진행합니다.", flush=True)
                    # 글로벌 모델을 클라이언트 모델과 동일하게 초기화
                    global_model.load_state_dict(client_model.state_dict())
                else:
                    raise e
        except Exception as e:
            print(f"❌ 글로벌 모델 다운로드/로드 실패: {e}", flush=True)
            print("🔄 로컬 모델 초기화로 진행합니다.", flush=True)
            # 글로벌 모델을 클라이언트 모델과 동일하게 초기화
            global_model.load_state_dict(client_model.state_dict())
        
        acc_before = evaluate_local_accuracy(client_model, train_loader, device)
        
        # 모델 상태 확인 (디버깅)
        print(f"학습 전 모델 상태 확인:")
        print(f"  - 모델 파라미터 수: {sum(p.numel() for p in client_model.parameters())}")
        print(f"  - 첫 번째 레이어 가중치 범위: {client_model.feature_extractor[0].weight.min().item():.4f} ~ {client_model.feature_extractor[0].weight.max().item():.4f}")
        
        # 2단계: 로컬 학습 수행
        print(f"🎓 2단계: 로컬 모델 학습 시작...", flush=True)
        training_start_time = time.time()
        accuracy = 0.0  # 기본값
        try:
            from improved_model import improved_client_update
            updated_model, avg_loss, epochs, num_samples, accuracy = improved_client_update(
                client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device, class_weights
            )
            print(f"✅ 개선된 학습 함수 사용 완료", flush=True)
        except Exception as e:
            print(f"개선된 학습 실패, 기본 버전 사용: {e}", flush=True)
            result = client_update_full(
                client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device,
                use_kd=False, use_fedprox=False, use_pruning=False  # 안정성을 위해 모든 고급 기능 비활성화
            )
            if len(result) == 4:
                updated_model, avg_loss, epochs, num_samples = result
                accuracy = 0.0  # 기본 함수는 정확도를 반환하지 않음
            else:
                updated_model, avg_loss, epochs, num_samples, accuracy = result
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
        
        # 학습 후 모델 상태 확인 (디버깅)
        print(f"학습 후 모델 상태 확인:")
        print(f"  - 모델 파라미터 수: {sum(p.numel() for p in updated_model.parameters())}")
        print(f"  - 첫 번째 레이어 가중치 범위: {updated_model.feature_extractor[0].weight.min().item():.4f} ~ {updated_model.feature_extractor[0].weight.max().item():.4f}")
        
        # 로컬 학습 완료된 모델을 그대로 사용 (FedAvg 원칙)
        print(f"=== 로컬 학습 완료 ===")
        print(f"로컬 학습된 모델 파라미터를 서버로 전송할 준비 완료")
        
        # 학습된 모델을 클라이언트 모델에 복사
        client_model.load_state_dict(updated_model.state_dict())
        
        # === 3단계: 클라이언트 데이터를 CKKS로 암호화 ===
        encryption_start_time = time.time()
        print(f"\n🔐 3단계: 클라이언트 데이터 CKKS 암호화", flush=True)
        state_dict = client_model.state_dict()
        print(f"📦 모델 파라미터 수: {len(state_dict)}개 레이어", flush=True)
        
        # 1) Tensor → flat numpy vector
        flat = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])
        print(f"평면화된 벡터 크기: {len(flat)}")
        
        # 2) CKKS 암호화
        print(f"🔒 CKKS 암호화 진행 중...", flush=True)
        c0_list, c1_list = batch_encrypt(flat)
        encrypted_flat = {'c0_list': c0_list, 'c1_list': c1_list}
        encryption_end_time = time.time()
        encryption_duration = encryption_end_time - encryption_start_time
        print(f"✅ CKKS 암호화 완료 (소요시간: {encryption_duration:.2f}초)", flush=True)
        
        # === 4단계: 암호화된 데이터를 서버로 전송 ===
        upload_start_time = time.time()
        print(f"\n📤 4단계: 암호화된 데이터 서버 전송", flush=True)
        
        # NaN/Inf 값을 안전한 값으로 변환하는 함수
        def safe_float(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0  # NaN/Inf를 0으로 대체
            return float(value)
        
        def safe_complex_to_float(complex_val):
            real_part = safe_float(complex_val.real)
            imag_part = safe_float(complex_val.imag)
            return [real_part, imag_part]
        
        # 암호화된 데이터를 JSON으로 직렬화 (안전한 변환)
        encrypted_data = {
            'client_id': CLIENT_ID,
            'round_id': r + 1,
            'c0_list': [[safe_complex_to_float(c) for c in c0] for c0 in c0_list],
            'c1_list': [[safe_complex_to_float(c) for c in c1] for c1 in c1_list],
            'original_size': len(flat),
            'num_samples': int(num_samples),
            'loss': safe_float(avg_loss),
            'accuracy': safe_float(accuracy)  # 라운드별 정확도 추가
        }
        
        print(f"JSON 직렬화 데이터 확인:", flush=True)
        print(f"  loss: {encrypted_data['loss']}", flush=True)
        print(f"  accuracy: {encrypted_data['accuracy']}", flush=True)
        print(f"  num_samples: {encrypted_data['num_samples']}", flush=True)
        print(f"  c0_list 길이: {len(encrypted_data['c0_list'])}", flush=True)
        print(f"  c1_list 길이: {len(encrypted_data['c1_list'])}", flush=True)
        
        try:
            print(f"🔄 서버로 라운드 {r+1} 데이터 전송 중...", flush=True)
            response = requests.post(f"{SERVER_URL}/aggregate", json=encrypted_data, timeout=60)
            if response.status_code == 200:
                upload_end_time = time.time()
                upload_duration = upload_end_time - upload_start_time
                print(f"✅ 서버 전송 완료 (소요시간: {upload_duration:.2f}초)", flush=True)
                
                server_response = response.json()
                print(f"📋 서버 응답: {server_response}", flush=True)
                
                # 서버에서 다음 라운드 진행 허용 여부 확인
                if server_response.get("status") == "success":
                    print(f"✅ 라운드 {r+1} 집계 완료, 다음 라운드 진행 가능", flush=True)
                else:
                    print(f"⚠️ 서버 집계 중 문제 발생: {server_response.get('message', '알 수 없는 오류')}", flush=True)
                
                # 잠시 대기 (서버 처리 시간 확보)
                if r < NUM_ROUNDS - 1:  # 마지막 라운드가 아닌 경우
                    print(f"⏳ 다음 라운드 준비를 위해 2초 대기...", flush=True)
                    time.sleep(2)
                    
            else:
                print(f"❌ 서버 전송 실패: {response.status_code}", flush=True)
                print(f"응답 내용: {response.text}", flush=True)
        except Exception as e:
            print(f"❌ 서버 통신 오류: {e}", flush=True)
            print("로컬 학습만 진행합니다.", flush=True)
        
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        print(f"\n🏁 === 라운드 {r+1}/{NUM_ROUNDS} 완료 (총 소요시간: {round_duration:.2f}초) ===", flush=True)
        print(f"📊 성과 요약:", flush=True)
        print(f"  🎯 학습 전 정확도: {acc_before:.2f}%", flush=True)
        print(f"  🎯 학습 후 정확도: {acc_after:.2f}%", flush=True)
        print(f"  📉 평균 손실: {avg_loss:.4f}", flush=True)
        print(f"  📁 학습 샘플 수: {num_samples:,}", flush=True)
        print(f"⏰ 완료 시간: {time.strftime('%H:%M:%S')}", flush=True)
        
        if r < NUM_ROUNDS - 1:
            print(f"⏳ 다음 라운드 준비 중...", flush=True)
        print("=" * 60, flush=True)

    print("=== 모든 라운드 완료 ===", flush=True)
    
    # 최종 예측 수행
    print("=== 최종 예측 수행 ===", flush=True)
    
    # 모델 테스트 (디버깅)
    print("=== 모델 테스트 ===")
    client_model.eval()
    with torch.no_grad():
        # 간단한 테스트 데이터 생성
        test_input = torch.randn(5, input_dim).to(device)
        test_output = client_model(test_input)
        test_probs = torch.softmax(test_output, dim=1)
        print(f"테스트 입력 형태: {test_input.shape}")
        print(f"테스트 출력 형태: {test_output.shape}")
        print(f"테스트 출력 샘플: {test_output[:3]}")
        print(f"테스트 확률 샘플: {test_probs[:3]}")
        print(f"테스트 확률 합계: {test_probs.sum(dim=1)[:3]}")
    
    try:
        # 원본 데이터 로드 (예측용)
        if input_file and os.path.exists(input_file):
            # 원본 데이터를 그대로 로드 (전처리하지 않음)
            original_df = pd.read_csv(input_file)
            print(f"원본 데이터 로드: {len(original_df)}행, {len(original_df.columns)}열")
        else:
            # 기본 데이터 파일 사용
            original_df = pd.read_csv('diabetic_data.csv')
            print(f"기본 데이터 파일 사용: {len(original_df)}행, {len(original_df.columns)}열")
        
        # 예측용 데이터 전처리 (학습과 동일한 특성 사용)
        df_for_prediction = original_df.copy()
        drop_cols = ['encounter_id', 'patient_nbr']
        if all(col in df_for_prediction.columns for col in drop_cols):
            df_for_prediction = df_for_prediction.drop(columns=drop_cols)
        if 'readmitted' in df_for_prediction.columns:
            df_for_prediction['readmitted'] = df_for_prediction['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
        
        # 학습과 동일한 특성 선택 (8개 고정 특성)
        fixed_features = [
            'admission_source_id', 'time_in_hospital', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses'
        ]
        
        # 사용 가능한 특성만 선택
        available_features = [col for col in fixed_features if col in df_for_prediction.columns]
        
        # 부족한 경우 다른 숫자형 특성 추가
        if len(available_features) < 8:
            numeric_cols = df_for_prediction.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'readmitted']
            remaining_cols = [col for col in numeric_cols if col not in available_features]
            available_features.extend(remaining_cols[:8-len(available_features)])
        
        # 정확히 8개 특성만 사용
        selected_features_for_prediction = available_features[:8]
        
        # 선택된 특성으로 데이터 준비
        X_pred = df_for_prediction[selected_features_for_prediction].values.astype('float32')
        print(f"예측용 데이터 준비: {X_pred.shape}")
        print(f"예측에 사용되는 특성: {selected_features_for_prediction}")
        
        # 스케일링 적용 (학습과 동일한 방식)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_pred_scaled = scaler.fit_transform(X_pred)
        X_pred_scaled = np.nan_to_num(X_pred_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"스케일링 후 데이터 형태: {X_pred_scaled.shape}")
        print(f"스케일링 후 NaN 개수: {np.isnan(X_pred_scaled).sum()}")
        
        # 예측 수행 (스케일링된 데이터 사용)
        probabilities, predictions, feature_importance = predict_diabetes_probability_with_explanation(client_model, 
            DataLoader(list(zip(X_pred_scaled, [0]*len(X_pred_scaled))), batch_size=64, shuffle=False), 
            selected_features_for_prediction, device)
        
        # 원본 데이터에 예측 결과 추가 (원본 형식 유지)
        result_df = original_df.copy()
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬 (선택사항)
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        success = save_results_to_excel(result_df, probabilities, predictions, feature_importance)
        
        if success:
            print("=== 학습 및 예측 완료 ===", flush=True)
            print(f"총 {len(result_df)}개 데이터에 대한 예측 완료", flush=True)
            print(f"당뇨병 예측: {sum(predictions)}개", flush=True)
            print(f"정상 예측: {len(predictions) - sum(predictions)}개", flush=True)
            print(f"평균 당뇨병 확률: {np.mean(probabilities):.4f}", flush=True)
            return True
        else:
            print("결과 저장 실패", flush=True)
            return False
            
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedHybrid 클라이언트')
    parser.add_argument('--input_file', type=str, help='입력 데이터 파일 경로')
    args = parser.parse_args()
    
    success = main(args.input_file)
    sys.exit(0 if success else 1) 