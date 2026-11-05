st.markdown('<div class="main-title">🧫 실제 미생물 발효 CSV 데이터 기반 수율 예측 웹앱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">실제 배양 공정 데이터셋을 로드하여 독립 변수 탐색, 회귀 모델 평가 및 수율을 예측합니다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Real CSV Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_real_csv_data():
    """CSV 데이터를 로드하고 필요한 6개 핵심 컬럼으로 매핑하는 함수"""
    possible_files = [
        "100_Batches_IndPenSim_Statistics_2.csv",
        "100_Batches_IndPenSim_Statistics.csv",
        "fermentation_data.csv"
    ]
    
    file_path = None
    for f in possible_files:
        if os.path.exists(f):
            file_path = f
            break
            
    if file_path is None:
        # 파일이 없을 경우 인라인 기본 데이터셋 생성
        np.random.seed(42)
        n_samples = 300
        temp = np.random.uniform(25.0, 40.0, n_samples)
        ph = np.random.uniform(5.5, 7.5, n_samples)
        glucose = np.random.uniform(10.0, 50.0, n_samples)
        rpm = np.random.uniform(100, 500, n_samples)
        do = np.random.uniform(20.0, 80.0, n_samples)
        
        # 실제 미생물 반응을 모사한 비선형 수율 공식
        yield_val = (
            -(temp - 33)**2 * 0.15 
            - (ph - 6.5)**2 * 8.0 
            + 0.5 * glucose - 0.005 * (glucose**2)
            + 0.03 * rpm 
            + 0.2 * do 
            + 30.0 
            + np.random.normal(0, 2.5, n_samples)
        )
        yield_val = np.maximum(yield_val, 0.5)
        
        return pd.DataFrame({
            'Temperature': temp,
            'pH': ph,
            'Glucose_Conc': glucose,
            'Agitation_Speed': rpm,
            'Dissolved_Oxygen': do,
            'Yield': yield_val
        })
        
    try:
        raw_df = pd.read_csv(file_path)
        
        # 컬럼 매핑 규칙 정의 (IndPenSim 및 일반 CSV 대응)
        column_map = {}
        for col in raw_df.columns:
            col_lower = col.lower()
            if 'temp' in col_lower or 't:k' in col_lower:
                column_map[col] = 'Temperature'
            elif 'ph' in col_lower:
                column_map[col] = 'pH'
            elif 'glucose' in col_lower or 'sugar' in col_lower or 'substrate' in col_lower or 's:g/l' in col_lower:
                column_map[col] = 'Glucose_Conc'
            elif 'agitat' in col_lower or 'rpm' in col_lower:
                column_map[col] = 'Agitation_Speed'
            elif 'dissolved' in col_lower or 'do2' in col_lower:
                column_map[col] = 'Dissolved_Oxygen'
            elif 'yield' in col_lower or 'penicillin concentration' in col_lower or 'p_offline' in col_lower or 'x_offline' in col_lower:
                column_map[col] = 'Yield'

        df = raw_df.rename(columns=column_map)
        
        # 캘빈(K) 온도를 섭씨(C)로 변환 (필요시)
        if 'Temperature' in df.columns:
            if df['Temperature'].mean() > 200:
                df['Temperature'] = df['Temperature'] - 273.15

        # 6개 필수 컬럼 보완
        required_cols = ['Temperature', 'pH', 'Glucose_Conc', 'Agitation_Speed', 'Dissolved_Oxygen', 'Yield']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Temperature': df[col] = 32.0
                elif col == 'pH': df[col] = 6.5
                elif col == 'Glucose_Conc': df[col] = 25.0
                elif col == 'Agitation_Speed': df[col] = 250.0
                elif col == 'Dissolved_Oxygen': df[col] = 50.0
                elif col == 'Yield': df[col] = 40.0

        # 라만 센서 데이터 등 숫자 이름의 무거운 컬럼 제거하고 6개만 남기기
        clean_df = df[required_cols].dropna().reset_index(drop=True)
        return clean_df
        
    except Exception as e:
        st.error(f"CSV 로딩 중 오류가 발생하여 기본 데이터셋으로 대체합니다: {e}")
        # 오류 발생시 더미 데이터 생성
        np.random.seed(42)
        n_samples = 200
        return pd.DataFrame({
            'Temperature': np.random.uniform(28, 38, n_samples),
            'pH': np.random.uniform(5.8, 7.2, n_samples),
            'Glucose_Conc': np.random.uniform(15, 45, n_samples),
            'Agitation_Speed': np.random.uniform(150, 450, n_samples),
            'Dissolved_Oxygen': np.random.uniform(30, 70, n_samples),
            'Yield': np.random.uniform(20, 60, n_samples)
        })

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="실제 바이오 공정 데이터 기반 수율 예측 시스템",
    page_icon="🧫",
    layout="wide"
)

# Custom Styling (Fixed parameter: unsafe_allow_html=True)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧫 실제 미생물 발효 CSV 데이터 기반 수율 예측 웹앱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">실제 배양 공정 데이터셋을 로드하여 독립 변수 탐색, 회귀 모델 평가 및 수율을 예측합니다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Real CSV Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_real_csv_data():
