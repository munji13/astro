import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit 페이지 설정
st.title("H-R Diagram with Stellar Evolution Path")
st.write("주계열성의 광도와 표면 온도를 입력하여 H-R 다이어그램 상의 위치와 진화 경로를 시각화합니다.")

# 사용자 입력
luminosity = st.number_input("광도 (태양 광도 단위, L☉)", min_value=1e-4, max_value=1e6, value=1.0, format="%.4f")
temperature = st.number_input("표면 온도 (Kelvin)", min_value=2000, max_value=40000, value=5800, step=100)

# 질량 추정 (질량-광도 관계: L ∝ M^3.5)
def estimate_mass(luminosity):
    return (luminosity ** (1/3.5))  # 태양 질량 단위

mass = estimate_mass(luminosity)
st.write(f"추정된 별의 질량: {mass:.2f} 태양 질량")

# H-R 다이어그램 생성 함수
def plot_hr_diagram(luminosity, temperature, mass):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # H-R 다이어그램 배경 설정
    sns.set(style="whitegrid")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2000)  # 온도는 왼쪽이 높음
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("Surface Temperature (K)")
    ax.set_ylabel("Luminosity (L☉)")
    ax.set_title("Hertzsprung-Russell Diagram")
    
    # 주계열성 영역 표시
    main_sequence_temp = np.logspace(np.log10(2000), np.log10(40000), 100)
    main_sequence_lum = 10 ** (3.5 * np.log10(main_sequence_temp / 5800))  # L ∝ T^3.5
    ax.plot(main_sequence_temp, main_sequence_lum, 'k-', label="Main Sequence", alpha=0.5)
    
    # 별의 현재 위치 표시
    ax.scatter([temperature], [luminosity], color='red', s=100, label="Current Star", zorder=10)
    
    # 진화 경로 생성
    if mass < 0.8:  # 저질량 별 (적색왜성 → 백색왜성)
        path_temp = [temperature, temperature * 0.8, 30000, 30000]  # 원시성 → 주계열 → 백색왜성
        path_lum = [luminosity * 1000, luminosity, luminosity * 0.01, 1e-4]
    elif mass < 8:  # 중간 질량 별 (태양 유사 → 적색거성 → 백색왜성)
        path_temp = [temperature * 1.5, temperature, temperature * 0.5, 10000, 30000]  # 원시성 → 주계열 → 적색거성 → 백색왜성
        path_lum = [luminosity * 1000, luminosity, luminosity * 100, luminosity * 10, 1e-4]
    else:  # 고질량 별 (주계열 → 초거성 → 초신성)
        path_temp = [temperature * 1.2, temperature, temperature * 0.3, 5000]  # 원시성 → 주계열 → 초거성 → 초신성
        path_lum = [luminosity * 1000, luminosity, luminosity * 1000, 1e5]
    
    # 진화 경로 표시
    ax.plot(path_temp, path_lum, 'b--', label="Evolutionary Path", alpha=0.7)
    
    # 범례 추가
    ax.legend()
    
    return fig

# H-R 다이어그램 표시
if st.button("H-R 다이어그램 생성"):
    fig = plot_hr_diagram(luminosity, temperature, mass)
    st.pyplot(fig)

# 설명
st.markdown("""
### 사용 방법
1. 광도(L☉)와 표면 온도(K)를 입력하세요.
2. 'H-R 다이어그램 생성' 버튼을 클릭하세요.
3. H-R 다이어그램에 별의 현재 위치(빨간 점)와 추정된 진화 경로(파란 점선)가 표시됩니다.

### 참고
- 주계열성은 H-R 다이어그램에서 대각선 띠에 위치합니다.
- 진화 경로는 별의 질량에 따라 다르며, 저질량/중간질량/고질량 별로 구분됩니다.
- 이 프로그램은 단순화된 모델을 사용하며, 실제 천문학적 시뮬레이션은 더 복잡합니다.
""")
