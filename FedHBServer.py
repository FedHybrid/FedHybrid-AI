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
import asyncio
import time
from typing import Dict, List, Optional

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

# 동시성 제어를 위한 변수들
updates_buffer: Dict[int, Dict[str, dict]] = {}  # round_id -> {client_id -> update_data}
global_accuracies = []
current_round = 0
round_start_time: Dict[int, float] = {}  # round_id -> start_time
ROUND_TIMEOUT = 300  # 5분 타임아웃
MIN_CLIENTS_PER_ROUND = 1  # 최소 클라이언트 수
MAX_WAIT_TIME = 60  # 최대 대기 시간 (초)
ROUND_CONFIG = {
    "min_clients": 1,      # 최소 클라이언트 수
    "max_clients": 50,     # 최대 클라이언트 수 (확장 가능)
    "target_clients": 10,  # 목표 클라이언트 수
    "adaptive_timeout": True  # 적응형 타임아웃 사용
}

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
    client_id: str  # 클라이언트 식별자 추가
    round_id: int   # 라운드 식별자 추가

class RoundConfigRequest(BaseModel):
    min_clients: Optional[int] = None
    max_clients: Optional[int] = None
    target_clients: Optional[int] = None
    max_wait_time: Optional[int] = None
    round_timeout: Optional[int] = None

@app.get("/get_model")
def get_model():
    if os.path.exists("global_model.pth"):
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")
    else:
        # 파일이 없으면 빈 모델을 저장하고 반환
        torch.save(global_model.state_dict(), "global_model.pth")
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

@app.get("/status")
def get_status():
    """서버 상태 및 현재 라운드 정보 반환"""
    current_time = time.time()
    active_rounds = {}
    
    for round_id, round_buffer in updates_buffer.items():
        elapsed_time = current_time - round_start_time.get(round_id, current_time)
        client_count = len(round_buffer)
        
        # 집계 조건 상태 계산
        aggregation_status = {
            "ready_for_aggregation": False,
            "reason": "",
            "conditions": {
                "target_clients_reached": client_count >= ROUND_CONFIG["target_clients"],
                "max_wait_time_exceeded": elapsed_time > MAX_WAIT_TIME and client_count >= ROUND_CONFIG["min_clients"],
                "long_timeout_exceeded": elapsed_time > ROUND_TIMEOUT and client_count >= ROUND_CONFIG["min_clients"],
                "max_clients_reached": client_count >= ROUND_CONFIG["max_clients"]
            }
        }
        
        # 집계 준비 상태 확인
        if client_count >= ROUND_CONFIG["target_clients"]:
            aggregation_status["ready_for_aggregation"] = True
            aggregation_status["reason"] = f"목표 클라이언트 수 도달 ({client_count}/{ROUND_CONFIG['target_clients']})"
        elif elapsed_time > MAX_WAIT_TIME and client_count >= ROUND_CONFIG["min_clients"]:
            aggregation_status["ready_for_aggregation"] = True
            aggregation_status["reason"] = f"최대 대기 시간 초과 ({elapsed_time:.1f}초)"
        elif elapsed_time > ROUND_TIMEOUT and client_count >= ROUND_CONFIG["min_clients"]:
            aggregation_status["ready_for_aggregation"] = True
            aggregation_status["reason"] = f"긴 타임아웃 초과 ({elapsed_time:.1f}초)"
        elif client_count >= ROUND_CONFIG["max_clients"]:
            aggregation_status["ready_for_aggregation"] = True
            aggregation_status["reason"] = f"최대 클라이언트 수 도달 ({client_count}/{ROUND_CONFIG['max_clients']})"
        
        active_rounds[round_id] = {
            "clients": list(round_buffer.keys()),
            "client_count": client_count,
            "start_time": round_start_time.get(round_id, 0),
            "elapsed_time": elapsed_time,
            "aggregation_status": aggregation_status
        }
    
    return {
        "current_round": current_round,
        "round_config": ROUND_CONFIG,
        "timeouts": {
            "max_wait_time": MAX_WAIT_TIME,
            "round_timeout": ROUND_TIMEOUT
        },
        "active_rounds": active_rounds,
        "total_active_rounds": len(updates_buffer),
        "server_status": "running"
    }

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
        # 파일이 존재하는지 확인
        if not os.path.exists(output_filename):
            print(f"파일이 존재하지 않음: {output_filename}")
            return {"error": "파일이 생성되지 않았습니다"}
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_filename)
        print(f"파일 크기: {file_size} bytes")
        
        return FileResponse(
            output_filename, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=output_filename,
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
    except Exception as e:
        print(f"파일 다운로드 응답 실패: {e}")
        return {"error": f"파일 다운로드 실패: {str(e)}"}

@app.post("/config")
def update_round_config(request: RoundConfigRequest):
    """라운드 설정 동적 변경"""
    global ROUND_CONFIG, MAX_WAIT_TIME, ROUND_TIMEOUT
    
    changes = {}
    
    if request.min_clients is not None:
        if request.min_clients > 0 and request.min_clients <= ROUND_CONFIG["max_clients"]:
            ROUND_CONFIG["min_clients"] = request.min_clients
            changes["min_clients"] = request.min_clients
        else:
            return {"error": f"min_clients는 1 이상 {ROUND_CONFIG['max_clients']} 이하여야 합니다"}
    
    if request.max_clients is not None:
        if request.max_clients >= ROUND_CONFIG["min_clients"] and request.max_clients <= 100:
            ROUND_CONFIG["max_clients"] = request.max_clients
            changes["max_clients"] = request.max_clients
        else:
            return {"error": f"max_clients는 {ROUND_CONFIG['min_clients']} 이상 100 이하여야 합니다"}
    
    if request.target_clients is not None:
        if ROUND_CONFIG["min_clients"] <= request.target_clients <= ROUND_CONFIG["max_clients"]:
            ROUND_CONFIG["target_clients"] = request.target_clients
            changes["target_clients"] = request.target_clients
        else:
            return {"error": f"target_clients는 {ROUND_CONFIG['min_clients']} 이상 {ROUND_CONFIG['max_clients']} 이하여야 합니다"}
    
    if request.max_wait_time is not None:
        if 10 <= request.max_wait_time <= 600:
            MAX_WAIT_TIME = request.max_wait_time
            changes["max_wait_time"] = request.max_wait_time
        else:
            return {"error": "max_wait_time는 10초 이상 600초 이하여야 합니다"}
    
    if request.round_timeout is not None:
        if 60 <= request.round_timeout <= 1800:
            ROUND_TIMEOUT = request.round_timeout
            changes["round_timeout"] = request.round_timeout
        else:
            return {"error": "round_timeout는 60초 이상 1800초 이하여야 합니다"}
    
    return {
        "message": "설정이 성공적으로 업데이트되었습니다",
        "changes": changes,
        "current_config": {
            "round_config": ROUND_CONFIG,
            "max_wait_time": MAX_WAIT_TIME,
            "round_timeout": ROUND_TIMEOUT
        }
    }

@app.post("/aggregate")
async def aggregate(request: UpdateRequest):
    global global_model, updates_buffer, global_accuracies, current_round, round_start_time
    
    print(f"\n=== 서버: 클라이언트 {request.client_id} 암호화된 데이터 수신 (라운드 {request.round_id}) ===")
    print(f"받은 c0_list 길이: {len(request.c0_list)}")
    print(f"받은 c1_list 길이: {len(request.c1_list)}")
    print(f"원본 크기: {request.original_size}")
    print(f"샘플 수: {request.num_samples}")
    
    # 라운드별 버퍼 초기화
    if request.round_id not in updates_buffer:
        updates_buffer[request.round_id] = {}
        round_start_time[request.round_id] = time.time()
        print(f"새 라운드 {request.round_id} 버퍼 생성")
    
    # 현재 라운드 업데이트
    current_round = max(updates_buffer.keys())
    print(f"현재 진행 중인 라운드: {current_round}")
    print(f"활성 라운드들: {list(updates_buffer.keys())}")
    
    # 1) JSON → numpy array (암호화된 상태)
    c0_list = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in request.c0_list]
    c1_list = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in request.c1_list]
    
    print(f"암호화된 데이터 변환 완료: {len(c0_list)}개 배치")
    print(f"첫 번째 배치 c0 범위: {c0_list[0].min()} ~ {c0_list[0].max()}")
    print(f"첫 번째 배치 c1 범위: {c1_list[0].min()} ~ {c1_list[0].max()}")
    
    # 2) 클라이언트 업데이트 저장 (라운드별)
    updates_buffer[request.round_id][request.client_id] = {
        'c0_list': c0_list,
        'c1_list': c1_list,
        'num_samples': request.num_samples,
        'loss': request.loss,
        'timestamp': time.time()
    }
    
    print(f"클라이언트 {request.client_id} 업데이트 저장 완료 (라운드 {request.round_id})")
    print(f"라운드 {request.round_id} 버퍼 상태: {len(updates_buffer[request.round_id])}/{ROUND_CONFIG['target_clients']} 클라이언트 (목표)")
    print(f"라운드 {request.round_id} 대기 중인 클라이언트: {list(updates_buffer[request.round_id].keys())}")
    
    # 3) 타임아웃 체크
    current_time = time.time()
    if current_time - round_start_time[request.round_id] > ROUND_TIMEOUT:
        print(f"라운드 {request.round_id} 타임아웃 발생 ({ROUND_TIMEOUT}초)")
        # 타임아웃 시 현재까지 받은 업데이트로 집계 진행
    
    # 4) 적응형 라운드별 집계 조건 확인
    round_buffer = updates_buffer[request.round_id]
    elapsed_time = current_time - round_start_time[request.round_id]
    client_count = len(round_buffer)
    
    # 적응형 집계 조건 계산
    should_aggregate = False
    aggregation_reason = ""
    
    # 조건 1: 목표 클라이언트 수 도달
    if client_count >= ROUND_CONFIG["target_clients"]:
        should_aggregate = True
        aggregation_reason = f"목표 클라이언트 수 도달 ({client_count}/{ROUND_CONFIG['target_clients']})"
    
    # 조건 2: 최대 대기 시간 초과 (적응형)
    elif elapsed_time > MAX_WAIT_TIME and client_count >= ROUND_CONFIG["min_clients"]:
        should_aggregate = True
        aggregation_reason = f"최대 대기 시간 초과 ({elapsed_time:.1f}초 > {MAX_WAIT_TIME}초)"
    
    # 조건 3: 긴 타임아웃 (네트워크 문제 등)
    elif elapsed_time > ROUND_TIMEOUT and client_count >= ROUND_CONFIG["min_clients"]:
        should_aggregate = True
        aggregation_reason = f"긴 타임아웃 초과 ({elapsed_time:.1f}초 > {ROUND_TIMEOUT}초)"
    
    # 조건 4: 최대 클라이언트 수 도달
    elif client_count >= ROUND_CONFIG["max_clients"]:
        should_aggregate = True
        aggregation_reason = f"최대 클라이언트 수 도달 ({client_count}/{ROUND_CONFIG['max_clients']})"
    
    print(f"집계 조건 확인: {client_count}개 클라이언트, {elapsed_time:.1f}초 경과")
    print(f"집계 여부: {should_aggregate} ({aggregation_reason})")
    
    if should_aggregate:
        print(f"\n=== 서버: 라운드 {request.round_id} 암호화된 상태에서 평균 계산 시작 ===")
        print(f"집계 사유: {aggregation_reason}")
        print(f"라운드 {request.round_id}에 {len(round_buffer)}개 업데이트 있음")
        print(f"참여 클라이언트: {list(round_buffer.keys())}")
        
        # FedAvg 방식: 샘플 수 기반 가중치 계산
        client_weights = {}
        total_samples = sum(update['num_samples'] for update in round_buffer.values())
        
        for client_id, update in round_buffer.items():
            # 각 클라이언트의 샘플 수에 비례한 가중치
            weight = update['num_samples'] / total_samples
            client_weights[client_id] = weight
        
        print(f"클라이언트 가중치: {[(cid, f'{w:.3f}') for cid, w in client_weights.items()]}")
        
        # 암호화된 상태에서 가중 평균 계산 (복호화 없이)
        from ckks import ckks_add, ckks_scale
        
        # 첫 번째 클라이언트의 암호문을 가중치로 스케일링
        first_client_id = list(round_buffer.keys())[0]
        first_update = round_buffer[first_client_id]
        weight = client_weights[first_client_id]
        c0_list_agg = []
        c1_list_agg = []
        
        print(f"클라이언트 {first_client_id} 샘플 수: {first_update['num_samples']}")
        print(f"적용할 가중치: {weight:.4f}")
        
        for c0, c1 in zip(first_update['c0_list'], first_update['c1_list']):
            c0_scaled, c1_scaled = ckks_scale((c0, c1), weight)
            c0_list_agg.append(c0_scaled)
            c1_list_agg.append(c1_scaled)
        
        print(f"첫 번째 클라이언트 {first_client_id} (가중치: {weight:.3f})로 시작")
        
        # 단일 클라이언트인 경우 가중치만 적용
        if len(round_buffer) == 1:
            print(f"단일 클라이언트: 가중치 {weight:.3f} 적용 완료")
        else:
            # 나머지 클라이언트들과 암호화된 상태에서 가중 평균 계산
            for client_id, update in list(round_buffer.items())[1:]:
                weight = client_weights[client_id]
                print(f"클라이언트 {client_id} (가중치: {weight:.3f})와 가중 평균 계산")
                
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
        
        # 라운드 완료 처리
        print(f"라운드 {request.round_id} 완료")
        
        # 완료된 라운드 정리
        del updates_buffer[request.round_id]
        if request.round_id in round_start_time:
            del round_start_time[request.round_id]
        
        print(f"라운드 {request.round_id} 정리 완료")
        print(f"남은 활성 라운드: {list(updates_buffer.keys())}")
        
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