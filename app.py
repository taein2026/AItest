# app.py (UI/UX 개선 최종 버전)

import streamlit as st
from analysis import run_analysis # 1단계에서 만든 analysis.py 파일을 불러옵니다.
import pandas as pd # 총 진료 건수 계산을 위해 추가

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 이상치 탐지 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 사이드바 (파일 업로드 기능) ---
with st.sidebar:
    st.title("📄 파일 업로드")
    st.info("분석에 필요한 파일 3개를 모두 업로드해주세요.")
    
    main_file = st.file_uploader("① 메인 진료 데이터", type=['csv'])
    disease_file = st.file_uploader("② 상병명 매칭 테이블", type=['xlsx'])
    drug_file = st.file_uploader("③ 약물명 매칭 테이블", type=['xlsx'])
    
    # 모든 파일이 업로드 되면 분석 버튼 활성화
    st.markdown("---")
    start_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True, disabled=not(main_file and disease_file and drug_file))

# --- 메인 화면 ---
st.title("🤖 AI 의료기관 이상치 탐지 대시보드")
st.markdown("---")


# 1. 초기 화면 (파일 업로드 전)
if not (main_file and disease_file and drug_file):
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드한 후, '분석 시작하기' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_placeholder.png", use_column_width=True)


# 2. 분석 시작 (버튼 클릭 후)
if start_button:
    try:
        # 스피너와 함께 분석 시작
        with st.spinner('AI가 수만 건의 데이터를 분석하고 있습니다... (약 1~2분 소요)'):
            # 파일을 DataFrame으로 직접 읽어 총 건수 계산
            df_main_for_count = pd.read_csv(main_file, usecols=[0])
            total_claims = len(df_main_for_count)
            
            # 메인 분석 함수 실행
            results, fig = run_analysis(main_file, disease_file, drug_file)
        
        st.success("🎉 분석이 완료되었습니다! 아래 대시보드에서 결과를 확인하세요.")
        st.markdown("---")

        # 3. 분석 결과 대시보드
        
        # 3-1. 핵심 요약 지표 (Metric)
        col1, col2, col3 = st.columns(3)
        col1.metric("총 진료 건수", f"{total_claims:,} 건")
        col2.metric("탐지된 이상치", f"{len(results)} 건", f"{len(results)/total_claims:.2%}")
        col3.metric("분석된 특성(항목) 수", "500 개")
        
        st.markdown("---")

        # 3-2. 탭으로 결과 분리
        tab1, tab2 = st.tabs(["📊 이상치 요약 및 그래프", "📑 Top 20 상세 분석"])

        with tab1:
            st.header("이상치 분포 시각화")
            st.info("파란색 점들은 일반적인 진료 패턴을, 빨간색 점들은 특이 패턴(이상치)을 나타냅니다.")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("가장 의심스러운 진료 Top 20")
            st.info("Rank가 높을수록 패턴이 이질적이라는 의미입니다. 각 항목을 클릭하여 상세 원인을 확인하세요.")
            
            # Expander(확장 메뉴)를 사용해 각 결과를 깔끔하게 표시
            for res in reversed(results):
                with st.expander(f"**Rank {res['rank']}**: 환자번호 {res['patient_id']} (진료일: {res['date']})"):
                    st.write("▶ **이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):**")
                    st.dataframe(res['reasons'])
                    
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다. 로그를 확인해주세요: {e}")
