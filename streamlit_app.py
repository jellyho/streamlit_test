import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. 데이터 로드 및 전처리 ---
@st.cache_data # 데이터 캐싱
def load_data():
    """Iris 데이터를 로드하고 훈련/테스트셋으로 분할, 스케일링합니다."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # SGD는 스케일링이 매우 중요합니다.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test, y_train, y_test, class_names, X_test_scaled

X_train, X_test_orig, y_train, y_test, class_names, X_test = load_data()
all_classes = np.unique(y)

# --- 2. Streamlit 앱 UI ---

st.title("🚀 실시간 학습 과정 시각화 데모")
st.write("`SGDClassifier` (확률적 경사 하강법) 모델이 Epoch마다 똑똑해지는 과정을 지켜보세요.")

# --- 3. 사이드바: 모델 하이퍼파라미터 ---
st.sidebar.header("1. 모델 파라미터 설정")

n_epochs = st.sidebar.slider(
    "총 학습 횟수 (Epochs)", 
    min_value=10, 
    max_value=200, 
    value=50, 
    step=10
)

# 학습률 (Learning Rate)
learning_rate_init = st.sidebar.select_slider(
    "학습률 (Learning Rate)",
    options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    value=0.01
)

st.sidebar.info(
    """
    - **Epochs**: 전체 데이터셋을 몇 번 반복해서 학습할지 정합니다.
    - **Learning Rate**: 모델이 얼마나 '빠르게' 정답에 접근할지 보폭을 정합니다.
    
    **학습률이 너무 크면** 그래프가 심하게 널뛰고, 
    **너무 작으면** 학습이 매우 느리게 진행됩니다.
    """
)

st.sidebar.header("2. 데이터 확인")
if st.sidebar.checkbox("사용한 테스트 데이터 원본 보기"):
    st.sidebar.write(f"총 {len(X_test_orig)}개의 테스트 샘플:")
    st.sidebar.dataframe(pd.DataFrame(X_test_orig, columns=load_iris().feature_names))


# --- 4. 메인 화면: 학습 및 시각화 ---

st.header("모델 학습 및 실시간 결과")

# '모델 학습' 버튼
if st.button(f"{n_epochs} Epochs 동안 실시간 학습 시작!"):
    
    # (1) 모델 초기화
    # loss='log_loss'는 로지스틱 회귀와 유사하게 작동합니다.
    model = SGDClassifier(
        loss='log_loss', 
        max_iter=1,  # 1 Epoch씩 수동으로 제어할 것이므로 max_iter=1
        learning_rate='constant', 
        eta0=learning_rate_init, 
        random_state=42,
        warm_start=True # partial_fit을 사용하기 위해 True
    )
    
    # (2) 시각화를 위한 빈 공간(placeholder) 생성
    progress_bar = st.progress(0)
    status_text = st.empty() # 현재 Epoch 상태 텍스트
    chart_placeholder = st.empty() # 실시간 라인 차트
    
    history = [] # 정확도 기록

    # (3) 실시간 학습 루프
    for epoch in range(n_epochs):
        
        # 1 Epoch 학습
        model.partial_fit(X_train, y_train, classes=all_classes)
        
        # 테스트셋으로 성능 평가
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # 기록 추가
        history.append({'Epoch': epoch + 1, 'Test Accuracy': test_accuracy})
        history_df = pd.DataFrame(history).set_index('Epoch')
        
        # (4) UI 업데이트
        progress_bar.progress((epoch + 1) / n_epochs)
        status_text.text(f"Epoch {epoch + 1}/{n_epochs} | 현재 테스트 정확도: {test_accuracy:.4f}")
        chart_placeholder.line_chart(history_df)
        
        # 시각적 효과를 위해 아주 잠깐 멈춤
        time.sleep(0.05) 

    # (5) 학습 완료 후 최종 결과 표시
    st.success("🎉 모델 학습 완료!")
    st.balloons()
    
    st.subheader("최종 학습 결과")
    st.write(f"**최종 테스트 정확도:** {test_accuracy * 100:.2f}%")
    
    st.subheader("최종 혼동 행렬 (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    st.dataframe(cm_df)

else:
    st.info("사이드바에서 파라미터를 설정하고 '학습 시작' 버튼을 눌러주세요.")

