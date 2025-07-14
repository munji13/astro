import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

# Streamlit 페이지 설정
st.title("H-R Diagram Stellar Evolution Simulator")
st.write("Enter the star's luminosity (in solar units) and temperature (in Kelvin) to visualize its position and evolution on the H-R diagram.")

# 사용자 입력
luminosity = st.number_input("Luminosity (L☉)", min_value=0.0001, max_value=1000000.0, value=1.0, step=0.1)
temperature = st.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, value=5772.0, step=100.0)

# H-R 다이어그램 데이터 준비
def generate_hr_diagram(luminosity, temperature):
    # 기본 H-R 다이어그램 설정
    fig = go.Figure()

    # 주계열선 (Main Sequence) 예시 데이터
    main_sequence_temp = np.logspace(np.log10(3000), np.log10(40000), 100)
    main_sequence_lum = 10 ** (0.0004 * (main_sequence_temp - 5772) + np.log10(1.0))

    # 주계열선 플롯
    fig.add_trace(go.Scatter(
        x=main_sequence_temp,
        y=main_sequence_lum,
        mode='lines',
        name='Main Sequence',
        line=dict(color='blue')
    ))

    # 입력된 별의 위치
    fig.add_trace(go.Scatter(
        x=[temperature],
        y=[luminosity],
        mode='markers',
        name='Input Star',
        marker=dict(size=10, color='red', symbol='star')
    ))

    # 진화 경로 시뮬레이션 (간단한 모델)
    evolution_temp, evolution_lum = simulate_evolution(temperature, luminosity)

    # 진화 경로 플롯
    fig.add_trace(go.Scatter(
        x=evolution_temp,
        y=evolution_lum,
        mode='lines+markers',
        name='Evolution Track',
        line=dict(color='green', dash='dash'),
        marker=dict(size=5)
    ))

    # 그래프 레이아웃
    fig.update_layout(
        title="H-R Diagram",
        xaxis_title="Temperature (K)",
        yaxis_title="Luminosity (L☉)",
        xaxis=dict(autorange="reversed", type="log"),
        yaxis=dict(type="log"),
        showlegend=True,
        template="plotly_dark"
    )

    return fig

# 간단한 진화 경로 시뮬레이션 함수
def simulate_evolution(temperature, luminosity):
    # 초기 질량 추정 (주계열 근사)
    mass = estimate_mass(luminosity, temperature)
    
    # 진화 단계에 따른 온도와 광도 변화 (단순화된 모델)
    stages = 10
    evolution_temp = [temperature]
    evolution_lum = [luminosity]
    
    for i in range(stages):
        if mass > 8:  # 고질량 별
            # 거성 단계로 이동: 온도 감소, 광도 증가
            evolution_temp.append(evolution_temp[-1] * 0.9)
            evolution_lum.append(evolution_lum[-1] * 1.5)
        elif mass > 1:  # 중간 질량 별
            # 적색거성 단계: 온도 감소, 광도 증가
            evolution_temp.append(evolution_temp[-1] * 0.95)
            evolution_lum.append(evolution_lum[-1] * 1.2)
        else:  # 저질량 별
            # 적색거성 -> 백색왜성 경로
            evolution_temp.append(evolution_temp[-1] * 0.98)
            evolution_lum.append(evolution_lum[-1] * 0.8)
    
    return evolution_temp, evolution_lum

# 질량 추정 함수 (단순화된 모델)
def estimate_mass(luminosity, temperature):
    # 주계열 별의 광도-질량 관계 근사: L ~ M^3.5
    mass = (luminosity ** (1/3.5))
    return mass

# H-R 다이어그램 표시
fig = generate_hr_diagram(luminosity, temperature)
st.plotly_chart(fig, use_container_width=True)

# 진화 시뮬레이션 애니메이션
if st.button("Run Evolution Simulation"):
    st.write("Simulating stellar evolution...")
    evolution_temp, evolution_lum = simulate_evolution(temperature, luminosity)
    
    for i in range(len(evolution_temp)):
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=main_sequence_temp,
            y=main_sequence_lum,
            mode='lines',
            name='Main Sequence',
            line=dict(color='blue')
        ))
        fig_temp.add_trace(go.Scatter(
            x=[evolution_temp[i]],
            y=[evolution_lum[i]],
            mode='markers',
            name='Star Position',
            marker=dict(size=10, color='red', symbol='star')
        ))
        fig_temp.add_trace(go.Scatter(
            x=evolution_temp[:i+1],
            y=evolution_lum[:i+1],
            mode='lines',
            name='Evolution Track',
            line=dict(color='green', dash='dash')
        ))
        fig_temp.update_layout(
            title=f"H-R Diagram: Evolution Step {i+1}",
            xaxis_title="Temperature (K)",
            yaxis_title="Luminosity (L☉)",
            xaxis=dict(autorange="reversed", type="log"),
            yaxis=dict(type="log"),
            showlegend=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        time.sleep(0.5)

# 참고 문헌
st.write("### References")
st.write("- Stellar evolution tracks inspired by MESA (Modules for Experiments in Stellar Astrophysics).")
st.write("- H-R Diagram visualization using Plotly and Streamlit.")
