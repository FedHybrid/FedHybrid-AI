from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from advanced_model import AdvancedEnhancerModel, load_advanced_diabetes_data
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
import sys
from datetime import datetime
import shutil

# 로그 출력 개선을 위한 설정
def log_message(level: str, message: str):
    """타임스탬프와 로그 레벨을 포함한 로그 메시지 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    level_colors = {
        "INFO": "\033[94m",    # 파란색
        "WARNING": "\033[93m", # 노란색
        "ERROR": "\033[91m",   # 빨간색
        "SUCCESS": "\033[92m", # 초록색
        "DEBUG": "\033[90m"    # 회색
    }
    color = level_colors.get(level, "")
    reset = "\033[0m"
    
    log_line = f"[{timestamp}] {color}{level}{reset}: {message}"
    print(log_line, flush=True)  # 실시간 플러싱

def log_info(message: str):
    log_message("INFO", message)

def log_warning(message: str):
    log_message("WARNING", message)

def log_error(message: str):
    log_message("ERROR", message)

def log_success(message: str):
    log_message("SUCCESS", message)

def log_debug(message: str):
    log_message("DEBUG", message)

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js 클라이언트 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_info(f"디바이스 설정: {device}")

# CKKS 파라미터 설정 (클라이언트와 동일)
z_q = 1 << 10   # 2^10 = 1,024 (평문 인코딩용 스케일)
rescale_q = z_q  # 리스케일링용 스케일
N = 4  # 슬롯 수
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)  # 비밀키

# input_dim을 클라이언트와 반드시 동일하게 명시 (클라이언트는 8개 특성 사용)
input_dim = 8
num_classes = 2

# 클라이언트와 동일한 하이퍼파라미터 사용
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

global_model = AdvancedEnhancerModel(
    input_dim=input_dim, 
    num_classes=num_classes,
    hidden_dims=OPTIMIZED_PARAMS['hidden_dims'],
    dropout_rate=OPTIMIZED_PARAMS['dropout_rate']
).to(device)
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

# 서버 시작 시 global_model.pth가 있으면 로드, 없으면 저장 (호환성 체크 포함)
if os.path.exists("global_model.pth"):
    try:
        model_data = torch.load("global_model.pth", map_location=device, weights_only=False)
        
        # 새 형식 (메타데이터 포함)인지 확인
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            log_info(f"모델 메타데이터: {model_data.get('model_type', 'Unknown')} v{model_data.get('version', 'Unknown')}")
            log_info(f"입력 차원: {model_data.get('input_dim', 'Unknown')}")
            try:
                global_model.load_state_dict(model_data['state_dict'])
                log_success("기존 모델 로드 성공")
            except RuntimeError as e:
                log_warning(f"기존 모델과 호환되지 않습니다: {str(e)[:100]}...")
                log_info("새 모델로 초기화합니다")
                # 새 모델과 메타데이터 저장
                model_data = {
                    'state_dict': global_model.state_dict(),
                    'input_dim': 8,
                    'model_type': 'AdvancedEnhancerModel',
                    'version': '2.0'
                }
                torch.save(model_data, "global_model.pth")
                log_success("새 글로벌 모델 생성 및 저장 완료")
        else:
            # 구 형식 (state_dict만)
            log_warning("구 형식의 모델 파일입니다. 새 모델로 초기화합니다.")
            # 새 모델과 메타데이터 저장
            model_data = {
                'state_dict': global_model.state_dict(),
                'input_dim': 8,
                'model_type': 'AdvancedEnhancerModel',
                'version': '2.0'
            }
            torch.save(model_data, "global_model.pth")
            log_success("새 글로벌 모델 생성 및 저장 완료")
        
        log_success("기존 글로벌 모델 로드 완료")
    except (RuntimeError, KeyError) as e:
        log_warning(f"기존 모델과 호환되지 않습니다: {str(e)[:100]}...")
        log_info("기존 모델 파일을 백업하고 새 모델로 초기화합니다")
        
        # 기존 파일 백업
        backup_name = f"global_model_backup_{int(time.time())}.pth"
        import shutil
        shutil.move("global_model.pth", backup_name)
        log_info(f"기존 모델을 {backup_name}으로 백업했습니다")
        
        # 새 모델과 메타데이터 저장
        model_data = {
            'state_dict': global_model.state_dict(),
            'input_dim': 8,  # 고정된 입력 차원
            'model_type': 'AdvancedEnhancerModel',
            'version': '2.0'
        }
        torch.save(model_data, "global_model.pth")
        log_success("새 글로벌 모델 생성 및 저장 완료")
else:
    model_data = {
        'state_dict': global_model.state_dict(),
        'input_dim': 8,  # 고정된 입력 차원
        'model_type': 'AdvancedEnhancerModel',
        'version': '2.0'
    }
    torch.save(model_data, "global_model.pth")
    log_info("새 글로벌 모델 생성 및 저장 완료")

class UpdateRequest(BaseModel):
    c0_list: list
    c1_list: list
    original_size: int
    num_samples: int
    loss: float
    accuracy: float  # 라운드별 정확도 추가
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
        log_debug("글로벌 모델 파일 요청 - 기존 파일 반환")
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")
    else:
        # 파일이 없으면 빈 모델을 저장하고 반환
        log_warning("글로벌 모델 파일이 없어 새로 생성")
        model_data = {
            'state_dict': global_model.state_dict(),
            'input_dim': 8,  # 고정된 입력 차원
            'model_type': 'ImprovedEnhancerModel',
            'version': '1.0'
        }
        torch.save(model_data, "global_model.pth")
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

@app.post("/upload")
async def upload_client_update(request: UpdateRequest):
    """클라이언트 업데이트 수신 및 처리"""
    try:
        log_info(f"=== 클라이언트 {request.client_id} 암호화된 데이터 수신 (라운드 {request.round_id}) ===")
        log_debug(f"받은 c0_list 길이: {len(request.c0_list)}")
        log_debug(f"받은 c1_list 길이: {len(request.c1_list)}")
        log_debug(f"원본 크기: {request.original_size}")
        log_debug(f"샘플 수: {request.num_samples}")
        log_info(f"클라이언트 {request.client_id} 라운드 {request.round_id} 정확도: {request.accuracy}")
        log_info(f"클라이언트 {request.client_id} 라운드 {request.round_id} 손실: {request.loss}")
        
        # 라운드 버퍼 초기화 (필요한 경우)
        if request.round_id not in updates_buffer:
            log_info(f"새 라운드 {request.round_id} 버퍼 생성")
            updates_buffer[request.round_id] = {}
            round_start_time[request.round_id] = time.time()
        
        # 현재 진행 중인 라운드 업데이트
        global current_round
        current_round = max(current_round, request.round_id)
        log_debug(f"현재 진행 중인 라운드: {current_round}")
        log_debug(f"활성 라운드들: {list(updates_buffer.keys())}")
        
        # 암호화된 데이터 변환
        encrypted_data = []
        for i in range(len(request.c0_list)):
            c0 = np.array(request.c0_list[i], dtype=np.complex128)
            c1 = np.array(request.c1_list[i], dtype=np.complex128)
            encrypted_data.append((c0, c1))
        
        log_debug(f"암호화된 데이터 변환 완료: {len(encrypted_data)}개 배치")
        if len(encrypted_data) > 0:
            log_debug(f"첫 번째 배치 c0 범위: {encrypted_data[0][0].min()} ~ {encrypted_data[0][0].max()}")
            log_debug(f"첫 번째 배치 c1 범위: {encrypted_data[0][1].min()} ~ {encrypted_data[0][1].max()}")
        
        # 클라이언트 업데이트 저장
        updates_buffer[request.round_id][request.client_id] = {
            'encrypted_data': encrypted_data,
            'original_size': request.original_size,
            'num_samples': request.num_samples,
            'loss': request.loss,
            'accuracy': request.accuracy
        }
        
        log_success(f"클라이언트 {request.client_id} 업데이트 저장 완료 (라운드 {request.round_id})")
        log_info(f"라운드 {request.round_id} 버퍼 상태: {len(updates_buffer[request.round_id])}/{ROUND_CONFIG['target_clients']} 클라이언트 (목표)")
        log_debug(f"라운드 {request.round_id} 대기 중인 클라이언트: {list(updates_buffer[request.round_id].keys())}")
        
        # 집계 조건 확인
        elapsed_time = time.time() - round_start_time[request.round_id]
        client_count = len(updates_buffer[request.round_id])
        
        log_info(f"집계 조건 확인: {client_count}개 클라이언트, {elapsed_time:.1f} 초 경과")
        
        # 집계 여부 결정
        should_aggregate = (
            client_count >= ROUND_CONFIG["target_clients"] or
            (elapsed_time > MAX_WAIT_TIME and client_count >= ROUND_CONFIG["min_clients"]) or
            (elapsed_time > ROUND_TIMEOUT and client_count >= ROUND_CONFIG["min_clients"]) or
            client_count >= ROUND_CONFIG["max_clients"]
        )
        
        log_info(f"집계 여부: {should_aggregate} ()")
        
        return {"status": "success", "message": "업데이트 수신 완료", "should_aggregate": should_aggregate}
        
    except Exception as e:
        log_error(f"클라이언트 업데이트 처리 중 오류: {e}")
        return {"error": f"업데이트 처리 실패: {str(e)}"}

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
        original_df = pd.read_csv('diabetic_data.csv')
        print(f"원본 데이터 로드 완료: {len(original_df)}행, {len(original_df.columns)}열")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return {"error": "데이터 로드 실패"}
    
    # 2. 예측용 데이터 전처리 (별도로 처리)
    try:
        df_for_prediction = original_df.copy()
        # 클라이언트와 동일한 전처리 적용
        drop_cols = ['encounter_id', 'patient_nbr']
        if all(col in df_for_prediction.columns for col in drop_cols):
            df_for_prediction = df_for_prediction.drop(columns=drop_cols)
        if 'readmitted' in df_for_prediction.columns:
            df_for_prediction['readmitted'] = df_for_prediction['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
        
        # 숫자형 컬럼만 feature로 사용
        numeric_cols = df_for_prediction.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'readmitted']
        numeric_cols = [col for col in numeric_cols if col != 'max_glu_serum']
        
        X = df_for_prediction[numeric_cols].values.astype('float32')
        print(f"예측용 데이터 준비: {X.shape}")
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
                probabilities.extend(probs[:, 1].cpu().numpy())  # 당뇨병 확률 (클래스 1)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        print(f"예측 완료: {len(predictions)}개 샘플")
        print(f"예측 결과 분포: {np.bincount(predictions)}")
        print(f"평균 당뇨병 확률: {np.mean(probabilities):.4f}")
    except Exception as e:
        print(f"예측 실패: {e}")
        return {"error": "모델 예측 실패"}
    
    # 4. 원본 데이터에 예측 결과 추가 (원본 형식 유지)
    try:
        df_result = original_df.copy()
        df_result['당뇨병_확률'] = probabilities
        df_result['예측_결과'] = predictions
        df_result['예측_라벨'] = ['당뇨병' if p == 1 else '정상' for p in predictions]
        
        # 확률별로 정렬
        df_result = df_result.sort_values('당뇨병_확률', ascending=False)
        
        print(f"결과 데이터 생성 완료: {len(df_result)}행, {len(df_result.columns)}열")
    except Exception as e:
        print(f"결과 데이터 생성 실패: {e}")
        return {"error": "결과 데이터 생성 실패"}
    
    # 5. 엑셀 파일 저장
    try:
        output_filename = "diabetic_predictions.xlsx"
        
        # openpyxl 엔진을 사용하여 엑셀 파일 생성
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # 메인 결과 시트 (원본 데이터 + 예측 결과)
            df_result.to_excel(writer, sheet_name='예측결과', index=False)
            
            # 요약 통계 시트 추가
            summary_data = {
                '항목': ['총 데이터 수', '당뇨병 예측 수', '정상 예측 수', '평균 당뇨병 확률'],
                '값': [
                    len(df_result),
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
        
        print(f"엑셀 파일 저장 완료: {output_filename}")
    except Exception as e:
        print(f"엑셀 파일 저장 실패: {e}")
        import traceback
        traceback.print_exc()
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

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """데이터 파일 업로드 및 학습 시작"""
    try:
        print(f"\n=== 서버: 데이터 파일 업로드 시작 ===")
        print(f"업로드된 파일: {file.filename}")
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return {"error": "CSV 또는 Excel 파일만 업로드 가능합니다."}
        
        # 파일 저장
        file_path = f"uploaded_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"파일 저장 완료: {file_path}")
        
        # 기존 데이터 파일 백업 (있는 경우)
        if os.path.exists("diabetic_data.csv"):
            backup_path = f"diabetic_data_backup_{int(time.time())}.csv"
            os.rename("diabetic_data.csv", backup_path)
            print(f"기존 데이터 백업: {backup_path}")
        
        # 업로드된 파일을 diabetic_data.csv로 복사
        import shutil
        shutil.copy2(file_path, "diabetic_data.csv")
        print(f"데이터 파일 업데이트 완료: diabetic_data.csv")
        
        # 전역 모델 초기화 (새 데이터로 학습 시작)
        global global_model
        global_model = ImprovedEnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)
        model_data = {
            'state_dict': global_model.state_dict(),
            'input_dim': input_dim,
            'model_type': 'ImprovedEnhancerModel',
            'version': '1.0'
        }
        torch.save(model_data, "global_model.pth")
        print("전역 모델 초기화 완료")
        
        return {
            "message": "데이터 업로드 및 모델 초기화가 완료되었습니다.",
            "filename": file.filename,
            "file_path": file_path
        }
        
    except Exception as e:
        print(f"데이터 업로드 실패: {e}")
        return {"error": f"데이터 업로드 실패: {str(e)}"}

@app.post("/aggregate")
async def aggregate(request: UpdateRequest):
    global global_model, updates_buffer, global_accuracies, current_round, round_start_time
    
    log_info(f"=== 클라이언트 {request.client_id} 암호화된 데이터 수신 (라운드 {request.round_id}) ===")
    log_debug(f"받은 c0_list 길이: {len(request.c0_list)}")
    log_debug(f"받은 c1_list 길이: {len(request.c1_list)}")
    log_debug(f"원본 크기: {request.original_size}")
    log_debug(f"샘플 수: {request.num_samples}")
    log_info(f"클라이언트 {request.client_id} 라운드 {request.round_id} 정확도: {request.accuracy:.4f}")
    log_info(f"클라이언트 {request.client_id} 라운드 {request.round_id} 손실: {request.loss:.4f}")
    
    # 라운드별 버퍼 초기화
    if request.round_id not in updates_buffer:
        updates_buffer[request.round_id] = {}
        round_start_time[request.round_id] = time.time()
        log_info(f"새 라운드 {request.round_id} 버퍼 생성")
    
    # 현재 라운드 업데이트
    current_round = max(updates_buffer.keys())
    log_debug(f"현재 진행 중인 라운드: {current_round}")
    log_debug(f"활성 라운드들: {list(updates_buffer.keys())}")
    
    # 1) JSON → numpy array (암호화된 상태)
    c0_list = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in request.c0_list]
    c1_list = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in request.c1_list]
    
    log_debug(f"암호화된 데이터 변환 완료: {len(c0_list)}개 배치")
    log_debug(f"첫 번째 배치 c0 범위: {c0_list[0].min()} ~ {c0_list[0].max()}")
    log_debug(f"첫 번째 배치 c1 범위: {c1_list[0].min()} ~ {c1_list[0].max()}")
    
    # 2) 클라이언트 업데이트 저장 (라운드별)
    updates_buffer[request.round_id][request.client_id] = {
        'c0_list': c0_list,
        'c1_list': c1_list,
        'num_samples': request.num_samples,
        'loss': request.loss,
        'timestamp': time.time()
    }
    
    log_success(f"클라이언트 {request.client_id} 업데이트 저장 완료 (라운드 {request.round_id})")
    log_info(f"라운드 {request.round_id} 버퍼 상태: {len(updates_buffer[request.round_id])}/{ROUND_CONFIG['target_clients']} 클라이언트 (목표)")
    log_debug(f"라운드 {request.round_id} 대기 중인 클라이언트: {list(updates_buffer[request.round_id].keys())}")
    
    # 3) 타임아웃 체크
    current_time = time.time()
    if current_time - round_start_time[request.round_id] > ROUND_TIMEOUT:
        log_warning(f"라운드 {request.round_id} 타임아웃 발생 ({ROUND_TIMEOUT}초)")
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
    
    log_info(f"집계 조건 확인: {client_count}개 클라이언트, {elapsed_time:.1f}초 경과")
    log_info(f"집계 여부: {should_aggregate} ({aggregation_reason})")
    
    if should_aggregate:
        log_success(f"=== 라운드 {request.round_id} 암호화된 상태에서 평균 계산 시작 ===")
        log_info(f"집계 사유: {aggregation_reason}")
        log_info(f"라운드 {request.round_id}에 {len(round_buffer)}개 업데이트 있음")
        log_debug(f"참여 클라이언트: {list(round_buffer.keys())}")
        
        # FedAvg 방식: 샘플 수 기반 가중치 계산
        client_weights = {}
        total_samples = sum(update['num_samples'] for update in round_buffer.values())
        
        for client_id, update in round_buffer.items():
            # 각 클라이언트의 샘플 수에 비례한 가중치
            weight = update['num_samples'] / total_samples
            client_weights[client_id] = weight
        
        log_info(f"클라이언트 가중치: {[(cid, f'{w:.3f}') for cid, w in client_weights.items()]}")
        
        # 암호화된 상태에서 가중 평균 계산 (복호화 없이)
        from ckks import ckks_add, ckks_scale
        
        # 첫 번째 클라이언트의 암호문을 가중치로 스케일링
        first_client_id = list(round_buffer.keys())[0]
        first_update = round_buffer[first_client_id]
        weight = client_weights[first_client_id]
        c0_list_agg = []
        c1_list_agg = []
        
        log_debug(f"클라이언트 {first_client_id} 샘플 수: {first_update['num_samples']}")
        log_debug(f"적용할 가중치: {weight:.4f}")
        
        for c0, c1 in zip(first_update['c0_list'], first_update['c1_list']):
            c0_scaled, c1_scaled = ckks_scale((c0, c1), weight)
            c0_list_agg.append(c0_scaled)
            c1_list_agg.append(c1_scaled)
        
        log_info(f"첫 번째 클라이언트 {first_client_id} (가중치: {weight:.3f})로 시작")
        
        # 단일 클라이언트인 경우 가중치만 적용
        if len(round_buffer) == 1:
            log_success(f"단일 클라이언트: 가중치 {weight:.3f} 적용 완료")
        else:
            # 나머지 클라이언트들과 암호화된 상태에서 가중 평균 계산
            for client_id, update in list(round_buffer.items())[1:]:
                weight = client_weights[client_id]
                log_info(f"클라이언트 {client_id} (가중치: {weight:.3f})와 가중 평균 계산")
                
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
        
        log_success(f"암호화된 상태에서 평균 계산 완료")
        
        # 디버깅: 첫 번째 배치의 값 확인
        if len(c0_list_agg) > 0:
            first_c0 = c0_list_agg[0]
            first_c1 = c1_list_agg[0]
            log_debug(f"평균 계산된 첫 번째 배치 c0 범위: {first_c0.min()} ~ {first_c0.max()}")
            log_debug(f"평균 계산된 첫 번째 배치 c1 범위: {first_c1.min()} ~ {first_c1.max()}")
        
        # 라운드 완료 처리
        log_success(f"라운드 {request.round_id} 완료")
        
        # 완료된 라운드 정리
        del updates_buffer[request.round_id]
        if request.round_id in round_start_time:
            del round_start_time[request.round_id]
        
        log_info(f"라운드 {request.round_id} 정리 완료")
        log_debug(f"남은 활성 라운드: {list(updates_buffer.keys())}")
        
        # 5단계: 클라이언트로 암호화된 평균 결과 전송
        response = {
            "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list_agg],
            "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list_agg],
            "original_size": request.original_size
        }
        log_success(f"클라이언트로 암호화된 평균 결과 전송 완료")
        return response
    
    return {"status": "waiting"}

def save_global_model_atomic(model, path="global_model.pth"):
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)

if __name__ == "__main__":
    log_success("=== FedHybrid 서버 시작 ===")
    log_info(f"서버 주소: http://0.0.0.0:8000")
    log_info(f"CKKS 파라미터: N={N}, z_q={z_q}")
    log_info(f"모델 설정: input_dim={input_dim}, num_classes={num_classes}")
    log_info("서버가 시작되었습니다. 클라이언트 연결을 기다리는 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)