from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from model import EnhancerModel
from aggregation import CommunicationEfficientFedHB
import uvicorn
import os

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input_dim을 클라이언트와 반드시 동일하게 명시 (예: 11)
input_dim = 11
num_classes = 2

global_model = EnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)
fed = CommunicationEfficientFedHB()
updates_buffer = []
global_accuracies = []

# 서버 시작 시 global_model.pth가 있으면 로드, 없으면 저장
if os.path.exists("global_model.pth"):
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
else:
    torch.save(global_model.state_dict(), "global_model.pth")

@app.get("/get_model")
def get_model():
    return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

@app.post("/submit_update")
async def submit_update(file: UploadFile = File(...)):
    # 클라이언트 업데이트 파일 수신 및 버퍼에 저장
    update_path = f"update_{file.filename}"
    with open(update_path, "wb") as f:
        f.write(await file.read())
    update = torch.load(update_path, map_location=device)
    updates_buffer.append(update)
    os.remove(update_path)
    return {"status": "received"}

@app.post("/aggregate")
def aggregate():
    global global_model, updates_buffer, global_accuracies
    if not updates_buffer:
        return {"status": "no updates"}
    
    # 간단한 FedAvg 집계
    global_state = global_model.state_dict()
    total_samples = sum(update.get('num_samples', 1) for update in updates_buffer)
    
    # 모든 파라미터를 0으로 초기화 (float32 타입으로)
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
    
    # 가중 평균 계산
    for update in updates_buffer:
        weight = update.get('num_samples', 1) / total_samples
        if 'state_dict' in update:
            for key in global_state.keys():
                if key in update['state_dict']:
                    update_tensor = update['state_dict'][key].float()
                    global_state[key] += weight * update_tensor
        elif 'quant' in update:
            for key in global_state.keys():
                if key in update['quant']:
                    update_tensor = update['quant'][key].float()
                    global_state[key] += weight * update_tensor
    
    # 글로벌 모델 업데이트
    global_model.load_state_dict(global_state)
    
    # 집계 후 항상 파일로 저장
    save_global_model_atomic(global_model)
    updates_buffer = []
    
    return {"status": "aggregated"}

def save_global_model_atomic(model, path="global_model.pth"):
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)