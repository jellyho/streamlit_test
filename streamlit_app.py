import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- 1. 데이터 로드 및 전처리 (MNIST) ---
# st.cache_data: 이 함수는 한 번만 실행되고 결과가 캐시됩니다.
@st.cache_data
def load_data():
    """MNIST 데이터를 로드하고 훈련/테스트셋으로 분할, 스케일링합니다."""
    # st.spinner: 데이터 로딩 중에만 표시됩니다.
    with st.spinner("⏳ MNIST 데이터 로드 중... (최초 1회 시간이 걸릴 수 있습니다)"):
        # 70,000개의 (784,) 벡터 이미지 로드
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
        
        # y를 정수형으로 변환 (중요)
        y = y.astype(np.uint8)

        # MNIST의 표준 분할: 60,000 (학습) / 10,000 (테스트)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        # 시각화를 위해 원본 테스트 이미지 저장 (스케일링 전)
        X_test_orig = X_test.copy()

        # SGD를 위한 스케일링 (픽셀값 0-255 -> 표준 정규 분포)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # partial_fit에 필요한 전체 클래스 (0~9)
        all_classes = np.unique(y)
    
    return X_train, X_test, y_train, y_test, all_classes, X_test_orig

# 데이터 로드
X_train, X_test, y_train, y_test, all_classes, X_test_orig = load_data()


# --- 2. Streamlit 앱 UI ---
st.title("🚀 MNIST 실시간 학습 및 예측 시각화")
st.write("`SGDClassifier` 모델이 손글씨 숫자를 학습하는 과정을 실시간으로 확인하고, 결과를 직접 눈으로 보세요.")

# --- 3. 사이드바: 모델 하이퍼파라미터 ---
st.sidebar.header("1. 모델 파라미터 설정")

n_epochs = st.sidebar.slider(
    "총 학습 횟수 (Epochs)", 
    min_value=5, 
    max_value=100, 
    value=20, 
    step=5
)

learning_rate_init = st.sidebar.select_slider(
    "학습률 (Learning Rate)",
    options=[0.0001, 0.001, 0.01, 0.1],
    value=0.001
)

st.sidebar.info(
    """
    MNIST(784차원)는 Iris보다 훨씬 복잡합니다.
    - **Epochs**가 더 많이 필요합니다.
    - **Learning Rate**는 더 낮아야 안정적으로 수렴합니다.
    """
)

# --- 4. 메인 화면: 학습 및 시각화 ---
st.header("1. 모델 학습 (실시간)")

if st.button(f"{n_epochs} Epochs 동안 실시간 학습 시작!"):
    
    # (1) 모델 초기화
    model = SGDClassifier(
        loss='log_loss', # 로지스틱 회귀와 유사
        max_iter=1,  
        learning_rate='constant', 
        eta0=learning_rate_init, 
        random_state=42,
        warm_start=True 
    )
    
    # (2) 시각화를 위한 빈 공간(placeholder) 생성
    progress_bar = st.progress(0, text="학습 대기 중...")
    status_text = st.empty() 
    chart_placeholder = st.empty() 
    
    history = [] # 정확도 기록

    # (3) 실시간 학습 루프
    start_time = time.time()
    for epoch in range(n_epochs):
        
        # 1 Epoch 학습
        model.partial_fit(X_train, y_train, classes=all_classes)
        
        # 테스트셋으로 성능 평가 (매 Epoch마다 하면 느리므로 2번에 1번만)
        if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # 기록 추가
            history.append({'Epoch': epoch + 1, 'Test Accuracy': test_accuracy})
            history_df = pd.DataFrame(history).set_index('Epoch')
            
            # (4) UI 업데이트
            epoch_time = time.time() - start_time
            status_text.text(f"Epoch {epoch + 1}/{n_epochs} | 현재 테스트 정확도: {test_accuracy:.4f} | 경과 시간: {epoch_time:.2f}초")
            chart_placeholder.line_chart(history_df)
        
        progress_bar.progress((epoch + 1) / n_epochs, text=f"Epoch {epoch + 1}/{n_epochs} 학습 중...")
        
    
    # (5) 학습 완료 후 최종 결과 표시
    st.success(f"🎉 모델 학습 완료! (최종 정확도: {test_accuracy * 100:.2f}%)")
    st.balloons()
    
    # --- 5. 최종 예측 결과 시각화 (사용자 요청) ---
    st.header("2. 예측 결과 시각화")
    st.write(f"테스트 샘플 {len(y_test)}개에 대한 예측 결과입니다.")

    # 오류난 인덱스 찾기
    error_indices = np.where(y_pred != y_test)[0]
    st.subheader(f"😭 모델이 헷갈려하는 이미지 (오답 노트) - 총 {len(error_indices)}개 오류")

    if len(error_indices) > 0:
        # 5개 또는 오류 개수만큼 랜덤 샘플링
        num_to_show = min(5, len(error_indices))
        cols = st.columns(num_to_show)
        random_errors = np.random.choice(error_indices, size=num_to_show, replace=False)
        
        for i, img_index in enumerate(random_errors):
            with cols[i]:
                # 784 -> 28x28 이미지로 변환
                image = X_test_orig[img_index].reshape(28, 28)
                st.image(image, caption=f"테스트 샘플 #{img_index}", clamp=True) # clamp=True: 픽셀값 범위 유지
                st.info(f"**정답:** {y_test[img_index]}")
                st.error(f"**모델 예측:** {y_pred[img_index]}")
    else:
        st.success("대단합니다! 모든 테스트 샘플을 정확하게 맞췄습니다!")

    # 정상 분류 인덱스 찾기
    correct_indices = np.where(y_pred == y_test)[0]
    st.subheader(f"😄 정상적으로 분류된 이미지 (정답 노트) - 총 {len(correct_indices)}개 정답")
    
    if len(correct_indices) > 0:
        num_to_show = 5
        cols = st.columns(num_to_show)
        random_corrects = np.random.choice(correct_indices, size=num_to_show, replace=False)
        
        for i, img_index in enumerate(random_corrects):
            with cols[i]:
                image = X_test_orig[img_index].reshape(28, 28)
                st.image(image, caption=f"테스트 샘플 #{img_index}", clamp=True)
                st.success(f"**정답:** {y_test[img_index]}")
                st.caption(f"**모델 예측:** {y_pred[img_index]}")
else:
    st.info("사이드바에서 파라미터를 설정하고 '학습 시작' 버튼을 눌러주세요.")

