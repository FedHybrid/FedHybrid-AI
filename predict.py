import torch
import pandas as pd
import numpy as np
from model import EnhancerModel, load_diabetes_data

# 1. 복호화된 flat 파라미터 불러오기
m_vals = np.load('decrypted_params.npy')  # 복호화된 flat 파라미터 (예시)

# 2. 모델 준비
input_dim = 11  # diabetic_data.csv의 feature 개수
num_classes = 2
device = torch.device('cpu')
model = EnhancerModel(input_dim=input_dim, num_classes=num_classes).to(device)

# 3. 복호화된 파라미터를 모델에 로드
def load_flat_params_to_model(model, m_vals):
    ptr = 0
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        numel = param.numel()
        arr = torch.from_numpy(m_vals[ptr:ptr+numel].astype(np.float32)).view(param.size())
        state_dict[name].copy_(arr)
        ptr += numel
    model.load_state_dict(state_dict)

load_flat_params_to_model(model, m_vals)

# 4. 데이터 준비 (원본 DataFrame, test_loader)
df_raw = pd.read_csv('diabetic_data.csv')
_, test_dataset = load_diabetes_data('diabetic_data.csv')
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5. 예측 함수
def predict_with_model(model, data_loader, device='cpu'):
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            output = model(x)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1)
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy()[:, 1])  # 1번 클래스 확률
    return preds, probs

preds, probs = predict_with_model(model, test_loader, device=device)

# 6. test셋 인덱스 추출 (train_test_split의 shuffle=False가 아니면 순서가 섞임)
# load_diabetes_data에서 test셋 인덱스를 반환하지 않으므로, 직접 분할 필요
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetic_data.csv')
drop_cols = ['encounter_id', 'patient_nbr']
df = df.drop(columns=drop_cols)
df['readmitted'] = df['readmitted'].map(lambda x: 0 if x == 'NO' else 1)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'readmitted']
X = df[numeric_cols].values
y = df['readmitted'].values
_, test_indices = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)

# 7. 예측 결과를 원본 데이터와 합쳐서 엑셀로 저장
df_result = df_raw.iloc[test_indices].reset_index(drop=True)
df_result['predicted_label'] = preds
df_result['probability'] = probs
df_result.to_excel('prediction_results.xlsx', index=False)
print("엑셀 파일 저장 완료: prediction_results.xlsx")