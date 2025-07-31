import torch
import requests
from model import client_update_full, EnhancerModel, load_diabetes_data
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
NUM_ROUNDS = 10

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
    for _ in range(5):
        r = requests.get(f"{SERVER_URL}/get_model")
        with open("global_model.pth", "wb") as f:
            f.write(r.content)
        try:
            state_dict = torch.load("global_model.pth", map_location=device, weights_only=False)
            client_model.load_state_dict(state_dict)
            global_model.load_state_dict(state_dict)  # 글로벌 모델도 업데이트
            os.remove("global_model.pth")
            return
        except Exception as e:
            print(f"global_model.pth 로드 실패, 재시도... ({e})")
            time.sleep(1)
    raise RuntimeError("global_model.pth를 정상적으로 다운로드하지 못했습니다.")

def predict_diabetes_probability(model, data_loader, device):
    """당뇨병 확률 예측"""
    model.eval()
    probabilities = []
    predictions = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs[:, 1].cpu().numpy())  # 당뇨병 확률 (클래스 1)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(probabilities), np.array(predictions)

def save_results_to_excel(original_data, probabilities, predictions, output_path='prediction_results.xlsx'):
    """결과를 엑셀 파일로 저장"""
    try:
        # 원본 데이터에 예측 결과 추가
        result_df = original_data.copy()
        result_df['당뇨병_확률'] = probabilities
        result_df['예측_결과'] = predictions
        result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬
        result_df = result_df.sort_values('당뇨병_확률', ascending=False)
        
        # 엑셀 파일로 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
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
        return True
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
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
    client_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)
    global_model = EnhancerModel(input_dim=input_dim, num_classes=2).to(device)  # 글로벌 모델 추가

    print(f"=== {NUM_ROUNDS}라운드 학습 시작 ===")
    
    for r in range(NUM_ROUNDS):
        round_start_time = time.time()  # 라운드 시작 시간
        print(f"=== 라운드 {r+1} 시작 ===")
        
        try:
            download_global_model()
        except Exception as e:
            print(f"글로벌 모델 다운로드 실패: {e}")
            print("로컬 학습만 진행합니다.")
        
        acc_before = evaluate_local_accuracy(client_model, train_loader, device)
        
        # 로컬 학습 수행
        training_start_time = time.time()
        updated_model, avg_loss, epochs, num_samples = client_update_full(
            client_model, global_model, train_loader, nn.CrossEntropyLoss(), r, device,
            use_kd=False, use_fedprox=False, use_pruning=True  # 일시적 비활성화
        )
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        acc_after = evaluate_local_accuracy(updated_model, train_loader, device)
        
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
            'round': r + 1,
            'encrypted_params': {
                'c0_list': [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list],
                'c1_list': [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list],
                'original_size': len(flat)
            },
            'num_samples': num_samples,
            'training_loss': float(avg_loss),
            'local_accuracy_before': float(acc_before),
            'local_accuracy_after': float(acc_after),
            'training_duration': float(training_duration),
            'encryption_duration': float(encryption_duration)
        }
        
        try:
            response = requests.post(f"{SERVER_URL}/upload_model", json=encrypted_data, timeout=30)
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
    try:
        # 원본 데이터 로드 (예측용)
        if input_file and os.path.exists(input_file):
            df = pd.read_csv(input_file)
            # 데이터 전처리 (model.py의 load_diabetes_data와 동일)
            drop_cols = ['encounter_id', 'patient_nbr']
            if all(col in df.columns for col in drop_cols):
                df = df.drop(columns=drop_cols)
            if 'readmitted' in df.columns:
                df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'readmitted' in numeric_cols:
                numeric_cols.remove('readmitted')
            
            X_pred = df[numeric_cols].values.astype('float32')
            
            # 예측 수행
            probabilities, predictions = predict_diabetes_probability(client_model, 
                DataLoader(list(zip(X_pred, [0]*len(X_pred))), batch_size=64, shuffle=False), device)
            
            # 결과 저장
            result_df = df.copy()
            result_df['당뇨병_확률'] = probabilities
            result_df['예측_결과'] = predictions
            result_df['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
            
            success = save_results_to_excel(result_df, probabilities, predictions)
            
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
                
        else:
            print("입력 파일을 찾을 수 없어 예측을 건너뜁니다.")
            return True
            
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedHybrid 클라이언트')
    parser.add_argument('--input_file', type=str, help='입력 데이터 파일 경로')
    args = parser.parse_args()
    
    success = main(args.input_file)
    sys.exit(0 if success else 1) 