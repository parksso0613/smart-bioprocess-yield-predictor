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
