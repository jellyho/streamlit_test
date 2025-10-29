import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 데이터 로드 및 전처리 ---
@st.cache_data # 데이터를 캐시에 저장하여 매번 다시 로드하지 않도록 함
def load_data():
    """Iris 데이터를 로드하고 2개의 클래스만 선택하여 반환합니다."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # 로지스틱 회귀(이진 분류) 데모를 위해 2개의 클래스(0, 1)만 선택
    df_binary = df[df['target'].isin([0, 1])].copy()
    
    # target 이름을 문자로 변경 (선택 사항)
    df_binary['target_name'] = df_binary['target'].map({0: iris.target_names[0], 1: iris.target_names[1]})
    
    X = df_binary[iris.feature_names]
    y = df_binary['target']
    return X, y, df_binary

X, y, df_binary = load_data()


# --- Streamlit 앱 UI ---

st.title("🌷 간단한 로지스틱 회귀(Logistic Regression) 데모")
st.write("Iris 데이터셋의 두 개 클래스(setosa, versicolor)를 분류하는 모델을 학습시킵니다.")

# --- 사이드바: 모델 파라미터 설정 ---
st.sidebar.header("1. 모델 파라미터 설정")

# C: 로지스틱 회귀의 규제 파라미터 (값이 작을수록 규제가 강해짐)
C_parameter = st.sidebar.slider(
    "C (규제 강도, 값이 클수록 규제 약함)", 
    min_value=0.01, 
    max_value=10.0, 
    value=1.0, 
    step=0.01
)

# 테스트 데이터셋의 비율 설정
test_size = st.sidebar.slider(
    "테스트 데이터 비율", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.3, 
    step=0.05
)

st.sidebar.header("2. 데이터 확인")
if st.sidebar.checkbox("사용한 데이터 미리보기"):
    st.sidebar.write("총 {}개의 샘플 사용:".format(len(df_binary)))
    st.sidebar.dataframe(df_binary.head())


# --- 메인 화면 ---

st.header("모델 학습 및 결과")

# '모델 학습' 버튼
if st.button("모델 학습 시작하기"):
    with st.spinner("데이터를 분할하고 모델을 학습시키는 중입니다..."):
        # 1. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 2. 모델 초기화 (사용자 설정 C값 적용)
        model = LogisticRegression(C=C_parameter, random_state=42)
        
        # 3. 모델 학습
        model.fit(X_train, y_train)
        
        # 4. 예측
        y_pred = model.predict(X_test)
        
        # 5. 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=["예측: setosa", "예측: versicolor"], index=["실제: setosa", "실제: versicolor"])

    # 6. 결과 표시
    st.success("🎉 모델 학습 완료!")
    
    st.subheader(f"모델 정확도 (Accuracy): {accuracy * 100:.2f}%")
    
    st.subheader("혼동 행렬 (Confusion Matrix)")
    st.dataframe(cm_df)
    st.write("""
    - **혼동 행렬**은 모델이 얼마나 헷갈려하는지 보여줍니다.
    - **대각선 (왼쪽 위 -> 오른쪽 아래)**의 숫자가 높을수록 좋습니다. (정확히 맞춘 개수)
    - 대각선 **밖**의 숫자는 모델이 틀리게 예측한 개수입니다.
    """)

else:
    st.info("사이드바에서 파라미터를 조절한 후 '모델 학습 시작하기' 버튼을 눌러주세요.")
