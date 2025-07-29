from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from model import EnhancerModel
from aggregation import CommunicationEfficientFedHB
import uvicorn
import os
import numpy as np
from ckks import batch_encrypt, batch_decrypt
from pydantic import BaseModel

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CKKS 파라미터 설정 (클라이언트와 동일)
Delta = 2**6  # 스케일 팩터
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

@app.get("/get_model")
def get_model():
    if os.path.exists("global_model.pth"):
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")
    else:
        # 파일이 없으면 빈 모델을 저장하고 반환
        torch.save(global_model.state_dict(), "global_model.pth")
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

@app.post("/aggregate")
async def aggregate(request: UpdateRequest):
    global global_model, updates_buffer, global_accuracies
    
    print(f"\n=== 서버: 클라이언트 업데이트 수신 ===")
    print(f"받은 c0_list 길이: {len(request.c0_list)}")
    print(f"받은 c1_list 길이: {len(request.c1_list)}")
    print(f"원본 크기: {request.original_size}")
    print(f"샘플 수: {request.num_samples}")
    
    # 1) JSON → numpy array
    c0_list = [np.array([complex(c[0], c[1]) for c in c0], dtype=np.complex128) for c0 in request.c0_list]
    c1_list = [np.array([complex(c[0], c[1]) for c in c1], dtype=np.complex128) for c1 in request.c1_list]
    
    print(f"복호화 시작: {len(c0_list)}개 배치")
    
    # 2) CKKS 배치 복호화
    m_vals = batch_decrypt(c0_list, c1_list, request.original_size, batch_size=4)
    
    print(f"복호화 완료: {len(m_vals)}개 값")
    print(f"복호화된 값 범위: {m_vals.real.min():.4f} ~ {m_vals.real.max():.4f}")
    print(f"복호화된 값 평균: {m_vals.real.mean():.4f}")
    
    # 3) 복호화된 벡터 → torch tensor + 원래 모델 shape 복원
    ptr = 0
    unflat_state = {}
    for k, v in global_model.state_dict().items():
        numel = v.numel()
        arr = torch.from_numpy(
            m_vals[ptr:ptr+numel].astype(np.float32)
        ).view(v.size())
        unflat_state[k] = arr.to(device)
        ptr += numel
        print(f"파라미터 {k}: shape={v.size()}, 값 범위={arr.min().item():.4f}~{arr.max().item():.4f}")
    
    print(f"모델 파라미터 복원 완료: {len(unflat_state)}개 레이어")
    
    # 4) 집계 (평균)
    updates_buffer.append((unflat_state, request.num_samples))
    print(f"업데이트 버퍼 크기: {len(updates_buffer)}/{EXPECTED_CLIENTS}")
    
    if len(updates_buffer) >= EXPECTED_CLIENTS:
        print(f"\n=== 서버: 모델 집계 시작 ===")
        # state_dict 평균
        total_samples = sum(samples for _, samples in updates_buffer)
        avg_state = {}
        for k in unflat_state.keys():
            weighted_sum = torch.zeros_like(unflat_state[k])
            for state_dict, samples in updates_buffer:
                weight = samples / total_samples
                weighted_sum += weight * state_dict[k]
            avg_state[k] = weighted_sum
            print(f"집계된 {k}: 평균값={weighted_sum.mean().item():.4f}")
        
        global_model.load_state_dict(avg_state)
        updates_buffer.clear()
        
        print(f"=== 서버: 재암호화 시작 ===")
        # 5) 재암호화 (평균 모델)
        flat_avg = np.concatenate([
            avg_state[k].cpu().numpy().flatten() for k in avg_state
        ])
        m_avg = flat_avg.astype(np.complex128)
        c0_list_agg, c1_list_agg = batch_encrypt(m_avg, batch_size=4)
        
        print(f"재암호화 완료: {len(c0_list_agg)}개 배치")
        print(f"집계된 모델 크기: {len(m_avg)}")
        
        # 6) 클라이언트로 암호문 송신
        response = {
            "c0_list": [[[float(c.real), float(c.imag)] for c in c0] for c0 in c0_list_agg],
            "c1_list": [[[float(c.real), float(c.imag)] for c in c1] for c1 in c1_list_agg],
            "original_size": len(m_avg)
        }
        print(f"클라이언트로 응답 전송 완료")
        return response
    
    return {"status": "waiting"}

def save_global_model_atomic(model, path="global_model.pth"):
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)