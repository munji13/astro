import streamlit as st
import plotly.graph_objects as go
from stellar_evolution import plot_hr_diagram, simulate_evolution

st.title("별의 H-R 도 및 진화 시뮬레이션")

# 사용자 입력
st.header("별의 속성 입력")
luminosity = st.number_input("광도 (태양 광도 L☉ 단위)", min_value=0.0001, max_value=1000000.0, value=1.0, step=0.1)
temperature = st.number_input("표면 온도 (켈빈 K)", min_value=1000, max_value=50000, value=5772, step=100)

# H-R 도 표시
st.header("H-R 도")
fig = plot_hr_diagram(luminosity, temperature)
st.plotly_chart(fig)

# 진화 시뮬레이션
st.header("진화 경로 시뮬레이션")
if st.button("진화 시뮬레이션 실행"):
    anim_fig = simulate_evolution(luminosity, temperature)
    st.plotly_chart(anim_fig)
