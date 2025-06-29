# analysis.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

def run_analysis(main_file, disease_file, drug_file):
    """
    3개의 파일을 입력받아 이상치 분석을 수행하고,
    결과 데이터프레임과 그래프를 반환하는 함수.
    """
    # 2단계: 데이터 파일 불러오기
    df = pd.read_csv(main_file, encoding='cp949', low_memory=False)
    df_disease_map = pd.read_excel(disease_file, dtype={'상병코드': str})
    df_drug_map = pd.read_excel(drug_file, dtype={'연합회코드': str})

    # 3단계: 한글 해석 사전 만들기
    df_disease_map['상병코드'] = df_disease_map['상병코드'].str.strip()
    df_drug_map['연합회코드'] = df_drug_map['연합회코드'].str.strip()
    disease_dict = pd.Series(df_disease_map['표준상병명'].values, index=df_disease_map['상병코드']).to_dict()
    drug_dict = pd.Series(df_drug_map['연합회전용명'].values, index=df_drug_map['연합회코드']).to_dict()
    code_to_name_map = {**disease_dict, **drug_dict}

    # 4-6단계: AI 모델 학습 및 예측
    df.columns = df.columns.astype(str)
    all_columns = df.columns.tolist()
    features1_cols = all_columns[all_columns.index('F41.2'):all_columns.index('J30.4') + 1]
    features2_cols = all_columns[all_columns.index('AA254'):all_columns.index('648101420') + 1]
    final_feature_cols = features1_cols + features2_cols
    features = df[final_feature_cols].copy()
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(features)
    df['anomaly_score'] = model.decision_function(features)
    df['anomaly_prediction'] = model.predict(features)
    
    # 7단계: 결과 분석 및 가공
    anomalies = df[df['anomaly_prediction'] == -1].sort_values(by='anomaly_score')
    top_20_anomalies = anomalies.head(20)
    normal_profile = features.mean()
    
    results = []
    for index, row in top_20_anomalies.iterrows():
        patient_id = row.get('환자번호', 'N/A')
        treatment_date = row.get('진료일시', 'N/A')
        rank = len(top_20_anomalies) - top_20_anomalies.index.get_loc(index)
        
        anomaly_to_explain = features.loc[index]
        actual_treatments = anomaly_to_explain[anomaly_to_explain == 1]
        
        reason_df = pd.DataFrame({
            '코드': actual_treatments.index,
            '평균 사용률': normal_profile[actual_treatments.index]
        }).sort_values(by='평균 사용률', ascending=True)
        reason_df['특성명 (한글)'] = reason_df['코드'].map(code_to_name_map).fillna("알 수 없는 코드")
        
        results.append({
            "rank": rank,
            "patient_id": patient_id,
            "date": treatment_date,
            "reasons": reason_df.head(5)
        })

    # 8단계: 그래프 생성
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(features)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    df_pca['anomaly_prediction'] = df.loc[df_pca.index, 'anomaly_prediction'].astype(str)
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='anomaly_prediction',
                     title="AI가 탐지한 이상치 분포 시각화 (-1: 이상치, 1: 정상)",
                     color_discrete_map={'1': 'blue', '-1': 'red'},
                     opacity=0.7)
    
    return results, fig