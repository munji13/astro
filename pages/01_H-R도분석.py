import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit 페이지 설정
st.title("H-R Diagram with Stellar Evolution Path (10 Stars)")
st.write("10개의 별의 질량을 입력하여 H-R 다이어그램 상의 위치와 독립적인 진화 경로를 시각화합니다.")

# 사용자 입력 (10개 별의 질량, 범위 0.08~100 M☉)
masses = []
for i in range(10):
    mass = st.number_input(f"별 {i+1} 질량 (태양 질량 단위, M☉)", min_value=0.08, max_value=100.0, value=0.08 + i * 10.0, format="%.2f", key=f"mass_{i}")
    masses.append(mass)

# 광도와 온도 추정 (보정된 관계 적용)
def estimate_luminosity_and_temperature(mass):
    if mass <= 0.5:  # 저질량 별 (L ∝ M^2.3)
        luminosity = (mass / 0.5) ** 2.3
    elif mass <= 2.0:  # 태양 근처 (L ∝ M^4)
        luminosity = (mass / 1.0) ** 4.0
    else:  # 고질량 별 (L ∝ M^3.5, 보정)
        luminosity = (mass / 2.0) ** 3.5 * 10.0
    temperature = 5800 * (mass ** 0.505) * (1 + 0.1 * np.log10(mass + 1))  # 보정된 온도 관계
    return luminosity, temperature

luminosities = []
temperatures = []
for mass in masses:
    lum, temp = estimate_luminosity_and_temperature(mass)
    luminosities.append(lum)
    temperatures.append(temp)
    st.write(f"별 {masses.index(mass)+1}: 질량 {mass:.2f} M☉, 광도 {lum:.2f} L☉, 온도 {temp:.0f} K")

# 베지어 곡선 생성 함수
def bezier_curve(points, n_points=100):
    t = np.linspace(0, 1, n_points)
    n = len(points) - 1
    curve = np.zeros((n_points, 2))
    for i in range(n_points):
        t_i = t[i]
        for j in range(n + 1):
            coef = 1.0
            for k in range(j):
                coef *= (n - k) / (k + 1)
            coef *= (1 - t_i) ** (n - j) * (t_i ** j)
            curve[i] += coef * np.array(points[j])
    return curve[:, 0], curve[:, 1]

# H-R 다이어그램 생성 함수
def plot_hr_diagram(masses, luminosities, temperatures):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # H-R 다이어그램 배경 설정 (범위 확장)
    sns.set(style="whitegrid")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150000, 1500)  # 온도 범위 확장
    ax.set_ylim(1e-6, 1e10)    # 광도 범위 확장
    ax.set_xlabel("Surface Temperature (K)")
    ax.set_ylabel("Luminosity (L☉)")
    ax.set_title("Hertzsprung-Russell Diagram (10 Stars)")
    
    # 주계열선 데이터 (현실적인 보정)
    main_sequence_masses = np.logspace(-1, 2, 100)  # 0.1 ~ 100 M☉
    main_sequence_lum = []
    main_sequence_temp = []
    for m in main_sequence_masses:
        l, t = estimate_luminosity_and_temperature(m)
        main_sequence_lum.append(l)
        main_sequence_temp.append(t)
    ax.plot(main_sequence_temp, main_sequence_lum, 'k-', label="Main Sequence", alpha=0.5)
    
    # 각 별의 현재 위치 및 독립적인 진화 경로 표시
    for i in range(len(masses)):
        mass = masses[i]
        lum = luminosities[i]
        temp = temperatures[i]
        
        # 현재 별 위치 표시
        ax.scatter([temp], [lum], color=f'C{i}', s=100, label=f"Star {i+1} (Main Sequence)", zorder=10)
        
        # 진화 경로 제어점 동적 계산
        if mass <= 0.08:  # 갈색왜성
            past_points = [(temp * 1.05, lum * 10), (temp, lum)]
            future_points = [(temp, lum), (temp * 0.98, lum * 0.005)]
        elif mass < 0.8:  # 저질량 별
            past_points = [(temp * 1.15, lum * 300), (temp * 1.05, lum * 100), (temp, lum)]
            future_points = [(temp, lum), (temp * 0.85, lum * 0.03), (30000 * mass / 0.8, 1e-5 * mass / 0.8)]
        elif mass < 8:  # 중간 질량 별
            past_points = [(temp * 1.25, lum * 600), (temp * 1.1, lum * 200), (temp, lum)]
            future_points = [(temp, lum), (temp * 0.55, lum * 120), (10000 * mass / 8, lum * 15 * mass / 8), (30000, 1e-5 * mass / 8)]
        else:  # 고질량 및 초고질량 별 (8~100 M☉)
            past_points = [(temp * 1.2, lum * 1000), (temp * 1.05, lum * 500), (temp, lum)]
            future_points = [(temp, lum), (temp * 0.35, lum * 1500), (5000 * mass / 100, 1e7 * mass / 100)]
        
        # 베지어 곡선으로 경로 그리기
        past_temp, past_lum = bezier_curve(past_points)
        future_temp, future_lum = bezier_curve(future_points)
        
        # 과거 경로 (파란 계열 곡선, 화살표)
        ax.plot(past_temp, past_lum, f'C{i}--', alpha=0.7)
        ax.arrow(past_temp[-2], past_lum[-2], past_temp[-1] - past_temp[-2], past_lum[-1] - past_lum[-2],
                 color=f'C{i}', width=0.05, head_width=0.2, head_length=0.3, alpha=0.7)
        
        # 미래 경로 (녹색 계열 곡선, 화살표)
        ax.plot(future_temp, future_lum, f'C{i+5}--', alpha=0.7)
        ax.arrow(future_temp[-2], future_lum[-2], future_temp[-1] - future_temp[-2], future_lum[-1] - future_lum[-2],
                 color=f'C{i+5}', width=0.05, head_width=0.2, head_length=0.3, alpha=0.7)
    
    # 범례 추가 (각 별의 현재 위치만 표시)
    ax.legend()
    
    return fig

# H-R 다이어그램 표시
if st.button("H-R 다이어그램 생성"):
    fig = plot_hr_diagram(masses, luminosities, temperatures)
    st.pyplot(fig)

# 설명
st.markdown("""
### 사용 방법
1. 10개의 별의 질량(M☉)을 입력하세요 (범위: 0.08~100 M☉).
2. 'H-R 다이어그램 생성' 버튼을 클릭하세요.
3. H-R 다이어그램에 각 별의 위치(다양한 색상 점), 과거 진화 경로(해당 색상 곡선, 화살표), 미래 진화 경로(다른 색상 곡선, 화살표)가 표시됩니다.

### 참고
- 광도와 온도는 질량에 따라 보정된 관계(L ∝ M^2.3~4.0, T ∝ M^0.505)를 사용해 추정됩니다.
- 주계열선은 질량 범위(0.1~100 M☉)에 맞춰 현실적으로 조정되었습니다.
- 진화 경로는 각 별의 질량에 따라 독립적으로 계산되며, 베지어 곡선으로 부드럽게 표현되며, 화살표로 방향을 나타냅니다.
- 0.08 M☉ 이하(갈색왜성), 0.8~8 M☉(중간 질량), 8~100 M☉(고질량/초고질량) 별에 따라 경로가 다릅니다.
- 이 프로그램은 단순화된 모델을 사용하며, 실제 천문학적 시뮬레이션은 더 복잡합니다.
""")
