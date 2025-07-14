import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

# Streamlit 페이지 설정
st.title("H-R Diagram with Stellar Evolution Path")
st.write("별의 질량을 입력하여 H-R 다이어그램 상의 주계열성 위치와 진화 경로를 시각화합니다.")

# 사용자 입력
mass = st.number_input("질량 (태양 질량 단위, M☉)", min_value=0.1, max_value=50.0, value=1.0, format="%.2f")

# 광도와 온도 추정
def estimate_luminosity_and_temperature(mass):
    luminosity = mass ** 3.5  # L/L☉ ≈ (M/M☉)^3.5
    temperature = 5800 * (mass ** 0.5)  # T/T☉ ≈ (M/M☉)^0.5, 태양 온도 5800K 기준
    return luminosity, temperature

luminosity, temperature = estimate_luminosity_and_temperature(mass)
st.write(f"추정된 광도: {luminosity:.2f} L☉")
st.write(f"추정된 표면 온도: {temperature:.0f} K")

# 베지어 곡선 생성 함수
def bezier_curve(points, n_points=100):
    t = np.linspace(0, 1, n_points)
    n = len(points) - 1
    curve = np.zeros((n_points, 2))
    for i in range(n_points):
        for j in range(n + 1):
            coef = np.math.comb(n, j) * (1 - t[i]) ** (n - j) * t[i] ** j
            curve[i] += coef * np.array(points[j])
    return curve[:, 0], curve[:, 1]

# H-R 다이어그램 생성 함수
def plot_hr_diagram(mass, luminosity, temperature):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # H-R 다이어그램 배경 설정
    sns.set(style="whitegrid")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(50000, 2000)  # 온도는 왼쪽이 높음
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("Surface Temperature (K)")
    ax.set_ylabel("Luminosity (L☉)")
    ax.set_title("Hertzsprung-Russell Diagram")
    
    # 주계열성 영역 표시
    main_sequence_temp = np.logspace(np.log10(2000), np.log10(50000), 100)
    main_sequence_lum = 10 ** (3.5 * np.log10(main_sequence_temp / 5800))
    ax.plot(main_sequence_temp, main_sequence_lum, 'k-', label="Main Sequence", alpha=0.5)
    
    # 현재 별 위치 표시
    ax.scatter([temperature], [luminosity], color='red', s=100, label="Current Star (Main Sequence)", zorder=10)
    
    # 진화 경로 정의
    if mass < 0.8:  # 저질량 별
        past_points = [(temperature * 1.5, luminosity * 1000), (temperature, luminosity)]  # 원시성 → 주계열
        future_points = [(temperature, luminosity), (temperature * 0.8, luminosity * 0.01), (30000, 1e-4)]  # 주계열 → 백색왜성
    elif mass < 8:  # 중간 질량 별
        past_points = [(temperature * 1.5, luminosity * 1000), (temperature, luminosity)]  # 원시성 → 주계열
        future_points = [(temperature, luminosity), (temperature * 0.5, luminosity * 100), (10000, luminosity * 10), (30000, 1e-4)]  # 주계열 → 적색거성 → 백색왜성
    else:  # 고질량 별
        past_points = [(temperature * 1.2, luminosity * 1000), (temperature, luminosity)]  # 원시성 → 주계열
        future_points = [(temperature, luminosity), (temperature * 0.3, luminosity * 1000), (5000, 1e5)]  # 주계열 → 초거성 → 초신성
    
    # 베지어 곡선으로 경로 그리기
    past_temp, past_lum = bezier_curve(past_points)
    future_temp, future_lum = bezier_curve(future_points)
    
    # 과거 경로 (파란 곡선, 화살표)
    ax.plot(past_temp, past_lum, 'b--', label="Past Evolution", alpha=0.7)
    ax.arrow(past_temp[-2], past_lum[-2], past_temp[-1] - past_temp[-2], past_lum[-1] - past_lum[-2],
             color='blue', width=0.05, head_width=0.2, head_length=0.3, alpha=0.7)
    
    # 미래 경로 (녹색 곡선, 화살표)
    ax.plot(future_temp, future_lum, 'g--', label="Future Evolution", alpha=0.7)
    ax.arrow(future_temp[-2], future_lum[-2], future_temp[-1] - future_temp[-2], future_lum[-1] - future_lum[-2],
             color='green', width=0.05, head_width=0.2, head_length=0.3, alpha=0.7)
    
    # 범례 추가
    ax.legend()
    
    return fig

# H-R 다이어그램 표시
if st.button("H-R 다이어그램 생성"):
    fig = plot_hr_diagram(mass, luminosity, temperature)
    st.pyplot(fig)

# 설명
st.markdown("""
### 사용 방법
1. 별의 질량(M☉)을 입력하세요.
2. 'H-R 다이어그램 생성' 버튼을 클릭하세요.
3. H-R 다이어그램에 주계열성의 위치(빨간 점), 과거 진화 경로(파란 곡선, 화살표), 미래 진화 경로(녹색 곡선, 화살표)가 표시됩니다.

### 참고
- 광도와 온도는 질량-광도 관계(L ∝ M^3.5)와 질량-온도 관계(T ∝ M^0.5)를 사용해 추정됩니다.
- 진화 경로는 베지어 곡선으로 부드럽게 표현되며, 화살표로 방향을 나타냅니다.
- 저질량(< 0.8 M☉), 중간 질량(0.8–8 M☉), 고질량(> 8 M☉) 별에 따라 경로가 다릅니다.
- 이 프로그램은 단순화된 모델을 사용하며, 실제 천문학적 시뮬레이션은 더 복잡합니다.
""")
