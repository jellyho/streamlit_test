import streamlit as st

# 1. 앱 제목 설정
st.title("🎈 Streamlit 데모 존나 신기하군")

# 2. 텍스트 입력 (st.text_input)
st.header("1. 텍스트 입력")
name = st.text_input("이름을 입력하세요:", "Gemini")

# 입력받은 텍스트를 동적으로 출력
st.write(f"안녕하세요, **{name}**님! 반갑습니다.")

# 3. 숫자 입력 (st.slider)
st.header("2. 슬라이더")
number = st.slider("숫자를 선택하세요:", 0, 100, 25) # (최소, 최대, 기본값)

st.write(f"선택한 숫자는 **{number}**입니다.")

# 4. 버튼 (st.button)
st.header("3. 버튼")
if st.button("여기를 눌러보세요!"):
    # 버튼이 클릭되면 실행
    st.write("버튼이 클릭되었습니다! 🎉")
    st.balloons() # 풍선 애니메이션
else:
    st.write("버튼을 아직 누르지 않았습니다.")