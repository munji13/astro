import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("쌍성계 주위 행성 궤적 시뮬레이션")

# 쌍성계 매개변수 입력
st.sidebar.header("쌍성계 설정")
M1 = st.sidebar.slider("항성 1 질량 (태양질량 단위)", 0.1, 5.0, 1.0, 0.1)
M2 = st.sidebar.slider("항성 2 질량 (태양질량 단위)", 0.1, 5.0, 1.0, 0.1)
d = st.sidebar.slider("항성 간 거리 (천문단위, AU)", 0.5, 10.0, 2.0, 0.5)

# 행성 초기 조건
st.sidebar.header("행성 초기 조건")
x0 = st.sidebar.slider("행성 초기 x 위치 (AU)", -10.0, 10.0, 5.0, 0.5)
y0 = st.sidebar.slider("행성 초기 y 위치 (AU)", -10.0, 10.0, 0.0, 0.5)
vx0 = st.sidebar.slider("행성 초기 x 속도 (AU/yr)", -5.0, 5.0, 0.0, 0.1)
vy0 = st.sidebar.slider("행성 초기 y 속도 (AU/yr)", -5.0, 5.0, 2.0, 0.1)

# 중력 상수 (AU^3 / 태양질량 / 년^2 단위)
G = 4 * np.pi**2

# 항성 위치 계산 (질량 중심을 원점으로)
x1 = -M2 * d / (M1 + M2)
x2 = M1 * d / (M1 + M2)
y1, y2 = 0.0, 0.0

# 운동 방정식 정의
def equations(state, t):
    x, y, vx, vy = state
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)
    ax = -G * M1 * (x - x1) / r1**3 - G * M2 * (x - x2) / r2**3
    ay = -G * M1 * y / r1**3 - G * M2 * y / r2**3
    return [vx, vy, ax, ay]

# 시간 배열
t = np.linspace(0, 50, 1000)

# 초기 상태
initial_state = [x0, y0, vx0, vy0]

# 운동 방정식 적분
solution = odeint(equations, initial_state, t)

# 궤적 데이터
x, y = solution[:, 0], solution[:, 1]

# 시각화
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, 'b-', label='행성 궤적')
ax.plot(x1, y1, 'ro', markersize=10, label=f'항성 1 (M={M1})')
ax.plot(x2, y2, 'go', markersize=10, label=f'항성 2 (M={M2})')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_title('쌍성계 주위 행성 궤적')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

# Streamlit에 플롯 표시
st.pyplot(fig)

# 설명 텍스트
st.write("""
이 애플리케이션은 쌍성계 주위를 공전하는 행성의 궤적을 시뮬레이션합니다.  
- **항성 1, 2 질량**: 태양질량 단위로 조정하여 중력 영향을 변경합니다.  
- **항성 간 거리**: 두 항성 간 거리(AU)를 변경하여 궤적 모양에 영향을 줍니다.  
- **행성 초기 조건**: 행성의 초기 위치와 속도를 설정하여 다양한 궤적을 탐색할 수 있습니다.  
궤적은 50년 동안의 운동을 보여줍니다.
""")
