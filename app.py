import streamlit as st
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

st.markdown('<div class="main-title">🧫 실제 미생물 발휴 CSV 데이터 기반 수율 예측 웹앱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">실제 배양 공정 데이터셋을 로드하여 독립 변수 탐색, 회귀 모델 평가 및 수율을 예측합니다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Real CSV Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_real_csv_data():
    # 사용자가 업로드한 CSV 파일명 탐색
    possible_filenames = [
        "100_Batches_IndPenSim_Statistics_2.csv",
        "100_Batches_IndPenSim_Statistics.csv",
        "fermentation_data.csv",
        "indpensim.csv"
    ]
    
    df_raw = None
    loaded_name = None
    
    for fname in possible_filenames:
        if os.path.exists(fname):
            try:
                df_raw = pd.read_csv(fname)
                loaded_name = fname
                break
            except Exception:
                pass

    if df_raw is not None:
        st.toast(f"✅ '{loaded_name}' 파일 로드 성공!", icon="🎉")
        
        # IndPenSim 컬럼 매핑 테이블 (복잡한 컬럼명을 표준 컬럼명으로 변환)
        column_mapping = {
            'Temperature(T:K)': 'Temperature',
            'pH(pH:pH)': 'pH',
            'Substrate concentration(S:g/L)': 'Glucose_Conc',
            'Agitator RPM(RPM:RPM)': 'Agitation_Speed',
            'Dissolved oxygen concentration(DO2:mg/L)': 'Dissolved_Oxygen',
            'Penicillin concentration(P:g/L)': 'Yield',
            'Offline Penicillin concentration(P_offline:P(g L^{-1}))': 'Yield',
            # 기본 컬럼명 처리
            'Temperature': 'Temperature',
            'pH': 'pH',
            'Glucose_Conc': 'Glucose_Conc',
            'Agitation_Speed': 'Agitation_Speed',
            'Dissolved_Oxygen': 'Dissolved_Oxygen',
            'Yield': 'Yield'
        }
        
        # 파일 컬럼 중 매핑 가능한 항목 찾기
        rename_dict = {}
        for col in df_raw.columns:
            for k, v in column_mapping.items():
                if k.lower() in col.lower() and v not in rename_dict.values():
                    rename_dict[col] = v
                    
        df_mapped = df_raw.rename(columns=rename_dict)
        
        # 필요한 주요 컬럼만 추출 (무거운 센서 열 수백 개 자동 제외)
        target_cols = ['Temperature', 'pH', 'Glucose_Conc', 'Agitation_Speed', 'Dissolved_Oxygen', 'Yield']
        available_cols = [c for c in target_cols if c in df_mapped.columns]
        
        if len(available_cols) >= 4 and 'Yield' in available_cols:
            df = df_mapped[available_cols].dropna()
            
            # IndPenSim 데이터 온도(Kelvin) 단위 섭씨(Celsius)로 자동 변환 (예: 298.15K -> 25°C)
            if df['Temperature'].mean() > 200:
                df['Temperature'] = df['Temperature'] - 273.15
                
            # 대용량 시계열 데이터인 경우 무거우므로 배치/단위별 샘플링 (최대 2000행 추출)
            if len(df) > 2000:
                df = df.sample(n=2000, random_state=42).sort_index()
                
            return df

    # 2. 로컬 파일이 없거나 컬럼 해석 실패 시 예비 바이오 데이터셋 사용
    st.toast("🌐 기본 바이오 발효 데이터셋으로 구동됩니다.", icon="ℹ️")
    np.random.seed(42)
    n_samples = 300
    temp = np.random.normal(32, 2, n_samples)
    ph = np.random.normal(6.5, 0.4, n_samples)
    glucose = np.random.uniform(15, 45, n_samples)
    agitation = np.random.uniform(150, 400, n_samples)
    do = np.random.uniform(30, 80, n_samples)
    
    # 미생물 반응 반응속도론 모사 수율 산출
    yield_val = (
        15 + 
        3.5 * glucose / (5 + glucose) + 
        -2.1 * (temp - 33)**2 + 
        -15.0 * (ph - 6.5)**2 + 
        0.05 * agitation + 
        0.2 * do + 
        np.random.normal(0, 1.5, n_samples)
    )
    yield_val = np.clip(yield_val, 5, 85)

    df = pd.DataFrame({
        'Temperature': np.round(temp, 2),
        'pH': np.round(ph, 2),
        'Glucose_Conc': np.round(glucose, 2),
        'Agitation_Speed': np.round(agitation, 1),
        'Dissolved_Oxygen': np.round(do, 1),
        'Yield': np.round(yield_val, 2)
    })
    return df

df = load_real_csv_data()

# -----------------------------------------------------------------------------
# Model Training & Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models(data):
    feature_cols = ['Temperature', 'pH', 'Glucose_Conc', 'Agitation_Speed', 'Dissolved_Oxygen']
    X = data[feature_cols]
    y = data['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    metrics = {
        'LR_R2': r2_score(y_test, lr_pred),
        'LR_RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'RF_R2': r2_score(y_test, rf_pred),
        'RF_RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'y_test': y_test,
        'rf_pred': rf_pred,
        'feature_importances': rf_model.feature_importances_
    }
    return lr_model, rf_model, metrics

lr_model, rf_model, metrics = train_models(df)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Simulator Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("📌 메뉴 및 시뮬레이터 제어")
app_mode = st.sidebar.radio("원하는 단계를 선택하세요:", [
    "1. CSV 데이터 탐색 (EDA)",
    "2. 소질문 시각화 분석",
    "3. 머신러닝 모델링 & 평가",
    "4. 실시간 수율 시뮬레이터"
])

# -----------------------------------------------------------------------------
# Mode 1: Data Exploration (EDA)
# -----------------------------------------------------------------------------
if app_mode == "1. CSV 데이터 탐색 (EDA)":
    st.subheader("📊 실제 CSV 로드 데이터 탐색")
    st.markdown("수집한 CSV 데이터셋의 실제 데이터 구조와 통계량을 분석합니다.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("##### 📄 로드된 CSV 데이터 미리보기 (상위 10개 행)")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.write("##### 📈 주요 요약 통계량")
        st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)
        
    st.divider()
    st.write("##### 🌡️ 변수 간 상관관계 열지도 (Correlation Heatmap)")
    corr = df.corr()
    fig_corr = px.imshow(
        corr, 
        text_auto=".2f", 
        color_continuous_scale="Blues",
        title="독립 변수 및 종속 변수(Yield) 간 상관계수 분석"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# Mode 2: Visualization for Sub-questions
# -----------------------------------------------------------------------------
elif app_mode == "2. 소질문 시각화 분석":
    st.subheader("🔍 탐구 소질문별 시각화 분석")
    
    sub_q = st.selectbox("분석할 소질문을 선택하세요:", [
        "소질문 ①: 배양 온도가 수율에 미치는 영향",
        "소질문 ②: pH와 용존 산소량(DO)이 수율에 미치는 복합 효과",
        "소질문 ③: 포도당 농도에 따른 수율의 비선형적 변화",
        "소질문 ④: 수율 결정 핵심 변수 중요도 분석"
    ])
    
    if "소질문 ①" in sub_q:
        st.markdown("**분석:** CSV 데이터상 온도의 변화에 따라 미생물 수율이 변하는 관계를 나타냅니다.")
        fig = px.scatter(df, x='Temperature', y='Yield', color='Yield', color_continuous_scale='Viridis',
                         trendline="loess", title="온도(Temperature) vs 수율(Yield)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif "소질문 ②" in sub_q:
        st.markdown("**분석:** pH와 용존 산소량(DO)의 3차원 분포와 수율 간의 상호작용 관계입니다.")
        fig = px.scatter_3d(df, x='pH', y='Dissolved_Oxygen', z='Yield', color='Yield',
                            title="pH, 용존산소량(DO) 및 수율 간 3D 분포")
        st.plotly_chart(fig, use_container_width=True)
        
    elif "소질문 ③" in sub_q:
        st.markdown("**분석:** 기질인 포도당 농도가 미생물 수율 증가 및 한계점에 미치는 영향입니다.")
        fig = px.scatter(df, x='Glucose_Conc', y='Yield', color='pH',
                         title="포도당 농도(Glucose_Conc) vs 수율(Yield)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif "소질문 ④" in sub_q:
        st.markdown("**분석:** 랜덤 포레스트 머신러닝 모델이 계산한 독립 변수들의 기여도(Feature Importance)입니다.")
        feat_df = pd.DataFrame({
            'Feature': ['Temperature', 'pH', 'Glucose_Conc', 'Agitation_Speed', 'Dissolved_Oxygen'],
            'Importance': metrics['feature_importances']
        }).sort_values(by='Importance', ascending=True)
        
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                     title="랜덤 포레스트 모델의 독립 변수 중요도", color='Importance', color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Mode 3: Modeling & Evaluation
# -----------------------------------------------------------------------------
elif app_mode == "3. 머신러닝 모델링 & 평가":
    st.subheader("🤖 실제 CSV 기반 회귀 모델링 성능 평가")
    st.markdown("선형 회귀(Linear Regression)와 랜덤 포레스트 회귀(Random Forest Regressor)의 예측 성능 지표를 비교합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="선형 회귀 R² Score", value=f"{metrics['LR_R2']:.4f}")
        st.metric(label="선형 회귀 RMSE", value=f"{metrics['LR_RMSE']:.4f}")
    with col2:
        st.metric(label="랜덤 포레스트 R² Score", value=f"{metrics['RF_R2']:.4f}", delta=f"{metrics['RF_R2'] - metrics['LR_R2']:.4f}")
        st.metric(label="랜덤 포레스트 RMSE", value=f"{metrics['RF_RMSE']:.4f}", delta=f"{metrics['RF_RMSE'] - metrics['LR_RMSE']:.4f}", delta_color="inverse")
        
    st.divider()
    st.write("##### 🎯 실제 수율 vs 모델 예측 수율 비교 (Random Forest)")
    
    eval_df = pd.DataFrame({'Actual': metrics['y_test'], 'Predicted': metrics['rf_pred']})
    fig_eval = px.scatter(eval_df, x='Actual', y='Predicted', title="Actual Yield vs Predicted Yield")
    fig_eval.add_shape(type="line", x0=df['Yield'].min(), y0=df['Yield'].min(),
                      x1=df['Yield'].max(), y1=df['Yield'].max(),
                      line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_eval, use_container_width=True)

# -----------------------------------------------------------------------------
# Mode 4: Real-time Simulator
# -----------------------------------------------------------------------------
elif app_mode == "4. 실시간 수율 시뮬레이터":
    st.subheader("🎛️ 실제 데이터 학습 모델 기반 시뮬레이터")
    st.markdown("CSV 데이터로 학습된 머신러닝 모델을 활용해 반응기 조건별 수율을 예측합니다.")
    
    col_input, col_result = st.columns([1, 1])
    
    min_temp, max_temp = float(df['Temperature'].min()), float(df['Temperature'].max())
    min_ph, max_ph = float(df['pH'].min()), float(df['pH'].max())
    min_glu, max_glu = float(df['Glucose_Conc'].min()), float(df['Glucose_Conc'].max())
    min_agi, max_agi = float(df['Agitation_Speed'].min()), float(df['Agitation_Speed'].max())
    min_do, max_do = float(df['Dissolved_Oxygen'].min()), float(df['Dissolved_Oxygen'].max())

    with col_input:
        st.write("##### ⚙️ 배양 환경 조건 조절")
        in_temp = st.slider("온도 (Temperature, °C)", min_temp, max_temp, float(df['Temperature'].mean()))
        in_ph = st.slider("pH 농도", min_ph, max_ph, float(df['pH'].mean()))
        in_glucose = st.slider("포도당 농도 (Glucose, g/L)", min_glu, max_glu, float(df['Glucose_Conc'].mean()))
        in_agitation = st.slider("교반 속도 (Agitation, RPM)", min_agi, max_agi, float(df['Agitation_Speed'].mean()))
        in_do = st.slider("용존 산소량 (DO, %)", min_do, max_do, float(df['Dissolved_Oxygen'].mean()))
        
    with col_result:
        st.write("##### 🔮 예측 결과 (Predict Result)")
        input_data = pd.DataFrame({
            'Temperature': [in_temp],
            'pH': [in_ph],
            'Glucose_Conc': [in_glucose],
            'Agitation_Speed': [in_agitation],
            'Dissolved_Oxygen': [in_do]
        })
        
        pred_yield = rf_model.predict(input_data)[0]
        
        st.metric(label="예상 미생물 생산 수율 (Yield)", value=f"{pred_yield:.2f} g/L")
        
        # Gauge Chart Visualization
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_yield,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "예측 수율 상태 (g/L)"},
            gauge = {
                'axis': {'range': [df['Yield'].min(), df['Yield'].max()]},
                'bar': {'color': "#1E3A8A"},
                'steps': [
                    {'range': [df['Yield'].min(), df['Yield'].quantile(0.33)], 'color': "#FEE2E2"},
                    {'range': [df['Yield'].quantile(0.33), df['Yield'].quantile(0.66)], 'color': "#FEF3C7"},
                    {'range': [df['Yield'].quantile(0.66), df['Yield'].max()], 'color': "#D1FAE5"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
