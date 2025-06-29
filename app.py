# app.py

import streamlit as st
from analysis import run_analysis # 1단계에서 만든 analysis.py 파일을 불러옵니다.

# 웹 페이지 제목 설정
st.set_page_config(page_title="AI 의료기관 이상치 탐지 시스템")
st.title("AI 의료기관 이상치 탐지 시스템 🔎")

# 파일 업로드 위젯
st.header("1. 분석할 파일 3개를 업로드하세요.")
main_file = st.file_uploader("메인 진료 데이터 (매핑완료_진료마스터...)", type=['csv'])
disease_file = st.file_uploader("상병명 매칭 테이블 (상병코드_표준상병명...)", type=['xlsx'])
drug_file = st.file_uploader("약물명 매칭 테이블 (연합회코드_연합회전용명...)", type=['xlsx'])

# 분석 시작 버튼
if st.button("분석 시작하기"):
    if main_file and disease_file and drug_file:
        with st.spinner('AI가 데이터를 분석 중입니다... 잠시만 기다려주세요.'):
            # 1단계에서 만든 함수를 호출하여 분석 실행
            results, fig = run_analysis(main_file, disease_file, drug_file)
        
        st.success("AI 분석이 완료되었습니다!")
        
        # 분석 결과 출력
        st.header("2. AI가 탐지한 Top 20 이상 의심 진료 분석 결과")
        for res in reversed(results): # 순위를 20위부터 보여주기 위해 reversed 사용
            st.subheader(f"Rank {res['rank']}: 환자번호 {res['patient_id']} ({res['date']})")
            st.write("▶ 이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):")
            st.dataframe(res['reasons'])

        # 그래프 출력
        st.header("3. 전체 데이터 이상치 분포 시각화")
        st.plotly_chart(fig)
    else:
        st.error("분석을 위해 3개의 파일을 모두 업로드해야 합니다.")
