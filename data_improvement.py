#!/usr/bin/env python3
"""
데이터 품질 개선을 위한 스크립트
정확도 향상을 위한 데이터 전처리 및 특성 엔지니어링
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data_quality(csv_path):
    """데이터 품질 분석"""
    print("=== 데이터 품질 분석 ===")
    
    df = pd.read_csv(csv_path)
    
    # 기본 정보
    print(f"데이터 크기: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 결측값 분석
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent': missing_percent
    })
    print("\n결측값 분석:")
    print(missing_info[missing_info['Missing Count'] > 0])
    
    # 중복값 분석
    duplicates = df.duplicated().sum()
    print(f"\n중복 행 수: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # 타겟 변수 분포
    if 'readmitted' in df.columns:
        target_dist = df['readmitted'].value_counts()
        print(f"\n타겟 변수 분포:")
        print(target_dist)
        print(f"클래스 불균형 비율: {target_dist.max() / target_dist.min():.2f}:1")
    
    return df

def feature_engineering(df):
    """특성 엔지니어링"""
    print("\n=== 특성 엔지니어링 ===")
    
    # 원본 데이터 복사
    df_engineered = df.copy()
    
    # 1. 범주형 변수 인코딩
    categorical_cols = df_engineered.select_dtypes(include=['object']).columns
    print(f"범주형 변수: {list(categorical_cols)}")
    
    for col in categorical_cols:
        if col != 'readmitted':  # 타겟 변수 제외
            # One-hot 인코딩
            dummies = pd.get_dummies(df_engineered[col], prefix=col, drop_first=True)
            df_engineered = pd.concat([df_engineered, dummies], axis=1)
            df_engineered.drop(col, axis=1, inplace=True)
    
    # 2. 수치형 변수 변환
    numeric_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'readmitted']
    
    # 로그 변환 (양수 값에만)
    for col in numeric_cols:
        if df_engineered[col].min() > 0:
            df_engineered[f'{col}_log'] = np.log1p(df_engineered[col])
    
    # 제곱 변환
    for col in numeric_cols:
        df_engineered[f'{col}_squared'] = df_engineered[col] ** 2
    
    # 3. 상호작용 특성
    if len(numeric_cols) >= 2:
        # 가장 중요한 특성들 간의 상호작용
        important_cols = ['time_in_hospital', 'num_medications', 'number_diagnoses']
        available_cols = [col for col in important_cols if col in numeric_cols]
        
        if len(available_cols) >= 2:
            for i in range(len(available_cols)):
                for j in range(i+1, len(available_cols)):
                    col1, col2 = available_cols[i], available_cols[j]
                    df_engineered[f'{col1}_{col2}_interaction'] = df_engineered[col1] * df_engineered[col2]
    
    print(f"엔지니어링 후 특성 수: {df_engineered.shape[1]}")
    return df_engineered

def feature_selection(df, target_col='readmitted', method='mutual_info'):
    """특성 선택"""
    print(f"\n=== 특성 선택 ({method}) ===")
    
    # 타겟 변수 분리
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 숫자형 특성만 선택
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_cols]
    
    # 특성 선택 방법
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(20, X_numeric.shape[1]))
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(20, X_numeric.shape[1]))
    else:
        raise ValueError("지원되지 않는 특성 선택 방법")
    
    # 특성 선택 수행
    X_selected = selector.fit_transform(X_numeric, y)
    selected_features = X_numeric.columns[selector.get_support()].tolist()
    
    print(f"선택된 특성 수: {len(selected_features)}")
    print(f"선택된 특성: {selected_features}")
    
    # 특성 중요도 점수
    scores = selector.scores_
    feature_scores = list(zip(X_numeric.columns, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n특성 중요도 (상위 10개):")
    for feature, score in feature_scores[:10]:
        print(f"  {feature}: {score:.4f}")
    
    return X_selected, selected_features, feature_scores

def evaluate_feature_importance(df, target_col='readmitted'):
    """랜덤 포레스트를 사용한 특성 중요도 평가"""
    print("\n=== 랜덤 포레스트 특성 중요도 ===")
    
    # 타겟 변수 분리
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 숫자형 특성만 선택
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_cols]
    
    # 랜덤 포레스트 모델
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_numeric, y)
    
    # 특성 중요도
    feature_importance = list(zip(X_numeric.columns, rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("랜덤 포레스트 특성 중요도 (상위 15개):")
    for feature, importance in feature_importance[:15]:
        print(f"  {feature}: {importance:.4f}")
    
    # 교차 검증 정확도
    cv_scores = cross_val_score(rf, X_numeric, y, cv=5, scoring='accuracy')
    print(f"\n랜덤 포레스트 교차 검증 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return feature_importance

def create_improved_dataset(csv_path, output_path):
    """개선된 데이터셋 생성"""
    print("=== 개선된 데이터셋 생성 ===")
    
    # 1. 데이터 품질 분석
    df = analyze_data_quality(csv_path)
    
    # 2. 특성 엔지니어링
    df_engineered = feature_engineering(df)
    
    # 3. 특성 선택
    X_selected, selected_features, feature_scores = feature_selection(df_engineered)
    
    # 4. 특성 중요도 평가
    feature_importance = evaluate_feature_importance(df_engineered)
    
    # 5. 개선된 데이터셋 생성
    # 상위 15개 특성 선택
    top_features = [feature for feature, _ in feature_importance[:15]]
    df_improved = df_engineered[top_features + ['readmitted']]
    
    # 개선된 데이터셋 저장
    df_improved.to_csv(output_path, index=False)
    print(f"\n개선된 데이터셋 저장: {output_path}")
    print(f"최종 특성 수: {len(top_features)}")
    print(f"최종 특성: {top_features}")
    
    return df_improved, top_features

if __name__ == "__main__":
    # 개선된 데이터셋 생성
    input_file = "diabetic_data.csv"
    output_file = "diabetic_improved.csv"
    
    try:
        df_improved, top_features = create_improved_dataset(input_file, output_file)
        print(f"\n✅ 데이터 개선 완료!")
        print(f"원본 파일: {input_file}")
        print(f"개선된 파일: {output_file}")
    except Exception as e:
        print(f"❌ 데이터 개선 실패: {e}")
