from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from model import EnhancerModel, load_diabetes_data
from aggregation import CommunicationEfficientFedHB
import uvicorn
import os
import numpy as np
from ckks import batch_encrypt, batch_decrypt
from pydantic import BaseModel
import pandas as pd
from torch.utils.data import DataLoader

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CKKS 파라미터 설정 (클라이언트와 동일)
z_q = 1 << 10   # 2^10 = 1,024 (평문 인코딩용 스케일)
rescale_q = z_q  # 리스케일링용 스케일
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

# input_dim을 클라이언트와 반드시 동일하게 명시 (예: 11)
input_dim = 11
num_classes = 2

global_model = EnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)
fed = CommunicationEfficientFedHB()
updates_buffer = []
global_accuracies = []
EXPECTED_CLIENTS = 1  # 예상 클라이언트 수

# 서버 시작 시 global_model.pth가 있으면 로드, 없으면 저장
if os.path.exists("global_model.pth"):
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device, weights_only=False))
else:
    torch.save(global_model.state_dict(), "global_model.pth")

class UpdateRequest(BaseModel):
    c0_list: list
    c1_list: list
    original_size: int
    num_samples: int
    loss: float

@app.get("/get_model")
def get_model():
    if os.path.exists("global_model.pth"):
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")
    else:
        # 파일이 없으면 빈 모델을 저장하고 반환
        torch.save(global_model.state_dict(), "global_model.pth")
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

@app.get("/predict_and_download")
def predict_and_download():
    """
    학습된 모델로 전체 데이터셋에 대해 예측을 수행하고 
    원본 데이터에 예측 결과를 추가한 엑셀 파일을 생성하여 다운로드
    """
    print(f"\n=== 서버: 예측 및 엑셀 파일 생성 시작 ===")
    
    # 1. 원본 데이터 로드
    try:
        df = pd.read_csv('diabetic_data.csv')
        print(f"원본 데이터 로드 완료: {len(df)}행, {len(df.columns)}열")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return {"error": "데이터 로드 실패"}
    
    # 2. 데이터 전처리 (클라이언트와 동일한 방식)
    try:
        # 클라이언트와 동일한 전처리 적용
        drop_cols = ['encounter_id', 'patient_nbr']
        df_processed = df.drop(columns=drop_cols)
        df_processed['readmitted'] = df_processed['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
        
        # 숫자형 컬럼만 feature로 사용
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'readmitted']
        
        X = df_processed[numeric_cols].values.astype('float32')
        print(f"전처리 완료: {X.shape}")
    except Exception as e:
        print(f"전처리 실패: {e}")
        return {"error": "데이터 전처리 실패"}
    
    # 3. 모델 예측
    try:
        global_model.eval()
        predictions = []
        probabilities = []
        
        # 배치 단위로 예측 수행
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_X).to(device)
            
            with torch.no_grad():
                outputs = global_model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        print(f"예측 완료: {len(predictions)}개 샘플")
        print(f"예측 결과 분포: {np.bincount(predictions)}")
    except Exception as e:
        print(f"예측 실패: {e}")
        return {"error": "모델 예측 실패"}
    
    # 4. 원본 데이터에 예측 결과 추가
    try:
        df_result = df.copy()
        df_result['predicted_readmission'] = predictions
        df_result['readmission_probability'] = probabilities[:, 1]  # 재입원 확률
        df_result['prediction_confidence'] = np.max(probabilities, axis=1)
        
        # 예측 결과 해석 추가
        df_result['prediction_interpretation'] = df_result['predicted_readmission'].map({
            0: '재입원 위험 낮음',
            1: '재입원 위험 높음'
        })
        
        print(f"결과 데이터 생성 완료: {len(df_result)}행, {len(df_result.columns)}열")
    except Exception as e:
        print(f"결과 데이터 생성 실패: {e}")
        return {"error": "결과 데이터 생성 실패"}
    
    # 5. 엑셀 파일 저장
    try:
        output_filename = "diabetic_predictions.xlsx"
        df_result.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"엑셀 파일 저장 완료: {output_filename}")
    except Exception as e:
        print(f"엑셀 파일 저장 실패: {e}")
        return {"error": "엑셀 파일 저장 실패"}
    
    # 6. 파일 다운로드 응답
    try:
        return FileResponse(
            output_filename, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=output_filename
        )
    except Exception as e:
        print(f"파일 다운로드 응답 실패: {e}")
        return {"error": "파일 다운로드 실패"}

@app.post("/aggregate")
async def aggregate(request: UpdateRequest):
    global global_model, updates_buffer, global_accuracies
    
    print(f"\n=== 서버: 클라이언트 암호화된 데이터 수신 ===")
    print(f"받은 c0_list 길이: {len(request.c0_list)}")
    print(f"받은 c1_list 길이: {len(request.c1_list)}")
    print(f"원본 크기: {request.original_size}")
    print(f"샘플 수: {request.num_samples}")
    
    # 1) JSON → numpy array (암호화된 상태)
    c0_list = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in request.c0_list]
    c1_list = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in request.c1_list]
    
    print(f"암호화된 데이터 변환 완료: {len(c0_list)}개 배치")
    print(f"첫 번째 배치 c0 범위: {c0_list[0].min()} ~ {c0_list[0].max()}")
    print(f"첫 번째 배치 c1 범위: {c1_list[0].min()} ~ {c1_list[0].max()}")
    
    # 2) CKKS 배치 복호화 (필요시에만)
    # m_vals = batch_decrypt(c0_list, c1_list, request.original_size, batch_size=4)
    # print(f"복호화 완료: {len(m_vals)}개 값")
    # print(f"복호화된 값 범위: {m_vals.real.min():.4f} ~ {m_vals.real.max():.4f}")
    # print(f"복호화된 값 평균: {m_vals.real.mean():.4f}")
    
    # 3) 복호화된 벡터 → torch tensor + 원래 모델 shape 복원 (필요시에만)
    # ptr = 0
    # unflat_state = {}
    # for k, v in global_model.state_dict().items():
    #     numel = v.numel()
    #     arr = torch.from_numpy(
    #         m_vals[ptr:ptr+numel].astype(np.float32)
    #     ).view(v.size())
    #     unflat_state[k] = arr.to(device)
    #     ptr += numel
    #     print(f"파라미터 {k}: shape={v.size()}, 값 범위={arr.min().item():.4f}~{arr.max().item():.4f}")
    # 
    # print(f"모델 파라미터 복원 완료: {len(unflat_state)}개 레이어")
    
    # 4) 암호화된 데이터 저장 (기존 버퍼 초기화 후 새로 추가)
    updates_buffer = [{
        'c0_list': c0_list,
        'c1_list': c1_list,
        'num_samples': request.num_samples,
        'loss': request.loss
    }]
    print(f"업데이트 버퍼 초기화 및 새 업데이트 추가: {len(updates_buffer)}/{EXPECTED_CLIENTS}")
    
    if len(updates_buffer) >= EXPECTED_CLIENTS:
        print(f"\n=== 서버: 암호화된 상태에서 평균 계산 시작 ===")
        print(f"현재 버퍼에 {len(updates_buffer)}개 업데이트 있음")
        
        # 손실 기반 가중치 계산
        client_weights = []
        for update in updates_buffer:
            # 손실이 낮을수록 높은 가중치
            weight = 1.0 / (1.0 + update['loss'])
            client_weights.append(weight)
        
        # 가중치 정규화
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        print(f"클라이언트 가중치: {[f'{w:.3f}' for w in client_weights]}")
        
        # 암호화된 상태에서 가중 평균 계산 (복호화 없이)
        from ckks import ckks_add, ckks_scale
        
        # 첫 번째 클라이언트의 암호문을 가중치로 스케일링
        first_update = updates_buffer[0]
        weight = client_weights[0]
        c0_list_agg = []
        c1_list_agg = []
        
        print(f"클라이언트 손실: {first_update['loss']:.4f}")
        print(f"적용할 가중치: {weight:.4f}")
        
        for c0, c1 in zip(first_update['c0_list'], first_update['c1_list']):
            c0_scaled, c1_scaled = ckks_scale((c0, c1), weight)
            c0_list_agg.append(c0_scaled)
            c1_list_agg.append(c1_scaled)
        
        print(f"첫 번째 클라이언트 (가중치: {weight:.3f})로 시작")
        
        # 단일 클라이언트인 경우 가중치만 적용
        if len(updates_buffer) == 1:
            print(f"단일 클라이언트: 가중치 {weight:.3f} 적용 완료")
        else:
            # 나머지 클라이언트들과 암호화된 상태에서 가중 평균 계산
            for i, update in enumerate(updates_buffer[1:], 1):
                weight = client_weights[i]
                print(f"클라이언트 {i+1} (가중치: {weight:.3f})와 가중 평균 계산")
                
                # 각 배치별로 가중치 적용 후 덧셈 수행
                for j in range(len(c0_list_agg)):
                    # 가중치 적용
                    c0_scaled, c1_scaled = ckks_scale((update['c0_list'][j], update['c1_list'][j]), weight)
                    
                    # 암호화된 덧셈 수행
                    c0_sum, c1_sum = ckks_add(
                        (c0_list_agg[j], c1_list_agg[j]), 
                        (c0_scaled, c1_scaled)
                    )
                    c0_list_agg[j] = c0_sum
                    c1_list_agg[j] = c1_sum
        
        print(f"암호화된 상태에서 평균 계산 완료")
        
        # 디버깅: 첫 번째 배치의 값 확인
        if len(c0_list_agg) > 0:
            first_c0 = c0_list_agg[0]
            first_c1 = c1_list_agg[0]
            print(f"평균 계산된 첫 번째 배치 c0 범위: {first_c0.min()} ~ {first_c0.max()}")
            print(f"평균 계산된 첫 번째 배치 c1 범위: {first_c1.min()} ~ {first_c1.max()}")
        
        updates_buffer.clear()
        
        # 5단계: 클라이언트로 암호화된 평균 결과 전송
        response = {
            "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list_agg],
            "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list_agg],
            "original_size": request.original_size
        }
        print(f"클라이언트로 암호화된 평균 결과 전송 완료")
        return response
    
    return {"status": "waiting"}

def save_global_model_atomic(model, path="global_model.pth"):
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)