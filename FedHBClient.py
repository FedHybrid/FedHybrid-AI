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
                    state_dict = torch.load("global_model.pth", map_location=device, weights_only=False)
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
    """결과를 엑셀 파일로 저장"""
    try:
        # 원본 데이터에 예측 결과 추가
        result_df = original_data.copy()
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        # 엑셀 파일로 저장 (openpyxl 엔진 사용)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 메인 결과 시트 (원본 데이터 + 예측 결과)
            result_df.to_excel(writer, sheet_name='예측결과', index=False)
            
            # 요약 통계 시트 추가
            summary_data = {
                '항목': ['총 데이터 수', '당뇨병 예측 수', '정상 예측 수', '평균 당뇨병 확률'],
                '값': [
                    len(result_df),
                    sum(predictions),
                    len(predictions) - sum(predictions),
                    f"{np.mean(probabilities):.4f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='요약통계', index=False)
            
            # 특성 중요도 시트 추가
            if feature_importance:
                importance_data = []
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                    importance_data.append({
                        '특성명': feature,
                        '중요도': importance,
                        '중요도_순위': len(importance_data) + 1
                    })
                importance_df = pd.DataFrame(importance_data)
                importance_df.to_excel(writer, sheet_name='특성중요도', index=False)
            
            # 확률 구간별 분포
            prob_ranges = [
                (0.0, 0.2, '매우 낮음'),
                (0.2, 0.4, '낮음'),
                (0.4, 0.6, '보통'),
                (0.6, 0.8, '높음'),
                (0.8, 1.0, '매우 높음')
            ]
            
            range_data = []
            for low, high, label in prob_ranges:
                count = sum((probabilities >= low) & (probabilities < high))
                range_data.append({
                    '확률_구간': f"{low:.1f}-{high:.1f}",
                    '라벨': label,
                    '데이터_수': count,
                    '비율': f"{count/len(probabilities)*100:.1f}%"
                })
            
            range_df = pd.DataFrame(range_data)
            range_df.to_excel(writer, sheet_name='확률구간별분포', index=False)
            
            # 상세 분석 시트 추가
            detail_data = []
            for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
                risk_level = '매우 높음' if prob >= 0.8 else \
                           '높음' if prob >= 0.6 else \
                           '보통' if prob >= 0.4 else \
                           '낮음' if prob >= 0.2 else '매우 낮음'
                
                detail_data.append({
                    '환자_번호': i + 1,
                    '당뇨병_확률': f"{prob:.4f}",
                    '예측_결과': '당뇨병' if pred == 1 else '정상',
                    '위험도': risk_level,
                    '권장사항': '즉시 의료진 상담 권장' if prob >= 0.8 else \
                              '정기 검진 권장' if prob >= 0.6 else \
                              '생활습관 개선 권장' if prob >= 0.4 else \
                              '건강 관리 권장' if prob >= 0.2 else '정상 관리'
                })
            
            detail_df = pd.DataFrame(detail_data)
            detail_df.to_excel(writer, sheet_name='상세분석', index=False)
        
        print(f"결과가 {output_path}에 저장되었습니다.")
        print(f"총 {len(result_df)}개 데이터에 대한 예측 완료")
        print(f"당뇨병 예측: {sum(predictions)}개")
        print(f"정상 예측: {len(predictions) - sum(predictions)}개")
        print(f"평균 당뇨병 확률: {np.mean(probabilities):.4f}")
        print(f"엑셀 파일 크기: {os.path.getsize(output_path)} bytes")
        
        if feature_importance:
            print(f"특성 중요도 분석 완료 - 상위 3개 특성:")
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_importance[:3]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(input_file=None):
    """메인 실행 함수"""
    print("=== FedHybrid 클라이언트 시작 ===")
    
    # 입력 파일 처리
    if input_file and os.path.exists(input_file):
        print(f"입력 파일: {input_file}")
        data_file = input_file
    else:
        print("기본 데이터 파일 사용: diabetic_data.csv")
        data_file = 'diabetic_data.csv'
    
    # 데이터셋 준비 (train/test 분할)
    try:
        train_dataset, test_dataset = load_diabetes_data(data_file)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        input_dim = train_dataset.X.shape[1]
        print(f"데이터 로드 완료 - 입력 차원: {input_dim}")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return False

    # 모델 준비 (EnhancerModel)
    client_model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=2).to(device)
    global_model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=2).to(device)  # 글로벌 모델 추가

    print(f"=== {NUM_ROUNDS}라운드 학습 시작 ===")
    
    for r in range(NUM_ROUNDS):
        round_start_time = time.time()  # 라운드 시작 시간
        print(f"=== 라운드 {r+1} 시작 ===")
        
        try:
            state_dict = download_global_model()
            global_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"글로벌 모델 다운로드 실패: {e}")
            print("로컬 학습만 진행합니다.")
        
        acc_before = evaluate_local_accuracy(client_model, train_loader, device)
        
        # 모델 상태 확인 (디버깅)
        print(f"학습 전 모델 상태 확인:")
        print(f"  - 모델 파라미터 수: {sum(p.numel() for p in client_model.parameters())}")
        print(f"  - 첫 번째 레이어 가중치 범위: {client_model.feature_extractor[0].weight.min().item():.4f} ~ {client_model.feature_extractor[0].weight.max().item():.4f}")
        
        # 로컬 학습 수행
        training_start_time = time.time()
        updated_model, avg_loss, epochs, num_samples = client_update_full(
            client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device,
            use_kd=True, use_fedprox=True, use_pruning=False  # FedProx와 KD 활성화
        )
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
        
        # === 1단계: 클라이언트 데이터를 CKKS로 암호화 ===
        encryption_start_time = time.time()
        print(f"\n=== 1단계: 클라이언트 데이터 CKKS 암호화 ===")
        state_dict = client_model.state_dict()
        print(f"모델 파라미터 수: {len(state_dict)}개 레이어")
        
        # 1) Tensor → flat numpy vector
        flat = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])
        print(f"평면화된 벡터 크기: {len(flat)}")
        
        # 2) CKKS 암호화
        c0_list, c1_list = batch_encrypt(flat)
        encrypted_flat = {'c0_list': c0_list, 'c1_list': c1_list}
        encryption_end_time = time.time()
        encryption_duration = encryption_end_time - encryption_start_time
        print(f"CKKS 암호화 완료 (소요시간: {encryption_duration:.2f}초)")
        
        # === 2단계: 암호화된 데이터를 서버로 전송 ===
        upload_start_time = time.time()
        print(f"\n=== 2단계: 암호화된 데이터 서버 전송 ===")
        
        # 암호화된 데이터를 JSON으로 직렬화
        encrypted_data = {
            'client_id': CLIENT_ID,
            'round_id': r + 1,
            'c0_list': [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list],
            'c1_list': [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list],
            'original_size': len(flat),
            'num_samples': num_samples,
            'loss': float(avg_loss)
        }
        
        try:
            response = requests.post(f"{SERVER_URL}/aggregate", json=encrypted_data, timeout=30)
            if response.status_code == 200:
                upload_end_time = time.time()
                upload_duration = upload_end_time - upload_start_time
                print(f"서버 전송 완료 (소요시간: {upload_duration:.2f}초)")
                print(f"서버 응답: {response.json()}")
            else:
                print(f"서버 전송 실패: {response.status_code}")
                print(f"응답 내용: {response.text}")
        except Exception as e:
            print(f"서버 통신 오류: {e}")
            print("로컬 학습만 진행합니다.")
        
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        print(f"=== 라운드 {r+1} 완료 (총 소요시간: {round_duration:.2f}초) ===")
        print(f"  - 학습 전 정확도: {acc_before:.2f}%")
        print(f"  - 학습 후 정확도: {acc_after:.2f}%")
        print(f"  - 평균 손실: {avg_loss:.4f}")
        print(f"  - 학습 샘플 수: {num_samples}")
        print()

    print("=== 모든 라운드 완료 ===")
    
    # 최종 예측 수행
    print("=== 최종 예측 수행 ===")
    
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
        
        # 예측용 데이터 전처리 (별도로 처리)
        df_for_prediction = original_df.copy()
        drop_cols = ['encounter_id', 'patient_nbr']
        if all(col in df_for_prediction.columns for col in drop_cols):
            df_for_prediction = df_for_prediction.drop(columns=drop_cols)
        if 'readmitted' in df_for_prediction.columns:
            df_for_prediction['readmitted'] = df_for_prediction['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
        
        numeric_cols = df_for_prediction.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'readmitted' in numeric_cols:
            numeric_cols.remove('readmitted')
        if 'max_glu_serum' in numeric_cols:
            numeric_cols.remove('max_glu_serum')
        
        X_pred = df_for_prediction[numeric_cols].values.astype('float32')
        print(f"예측용 데이터 준비: {X_pred.shape}")
        
        # 예측에 사용되는 특성 이름만 추출
        feature_names_for_prediction = numeric_cols
        print(f"예측에 사용되는 특성: {feature_names_for_prediction}")
        
        # 예측 수행
        probabilities, predictions, feature_importance = predict_diabetes_probability_with_explanation(client_model, 
            DataLoader(list(zip(X_pred, [0]*len(X_pred))), batch_size=64, shuffle=False), 
            feature_names_for_prediction, device)
        
        # 원본 데이터에 예측 결과 추가 (원본 형식 유지)
        result_df = original_df.copy()
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬 (선택사항)
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        success = save_results_to_excel(result_df, probabilities, predictions, feature_importance)
        
        if success:
            print("=== 학습 및 예측 완료 ===")
            print(f"총 {len(result_df)}개 데이터에 대한 예측 완료")
            print(f"당뇨병 예측: {sum(predictions)}개")
            print(f"정상 예측: {len(predictions) - sum(predictions)}개")
            print(f"평균 당뇨병 확률: {np.mean(probabilities):.4f}")
            return True
        else:
            print("결과 저장 실패")
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