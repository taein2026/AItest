# app.py (최종 디자인 개선 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 이상치 탐지 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lottie 애니메이션 로드 함수 ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- 사이드바 ---
with st.sidebar:
    # Lottie 애니메이션 표시
    lottie_url = "https://lottie.host/e883236e-1335-4309-a185-11a518012e69/Tpde6s5V1C.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, speed=1, height=150, key="initial")

    st.title("📄 파일 업로드")
    st.info("분석에 필요한 파일 3개를 모두 업로드해주세요.")
    
    main_file = st.file_uploader("① 메인 진료 데이터", type=['csv'])
    disease_file = st.file_uploader("② 상병명 매칭 테이블", type=['xlsx'])
    drug_file = st.file_uploader("③ 약물명 매칭 테이블", type=['xlsx'])
    
    st.markdown("---")
    start_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True, disabled=not(main_file and disease_file and drug_file))

# --- 메인 화면 ---
st.title("🤖 AI 의료기관 이상치 탐지 대시보드")
st.markdown("---")

# 초기 화면
if not start_button:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드한 후, '분석 시작하기' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_professional.png", use_column_width=True)


# 분석 시작
if start_button:
    try:
        # "생각하는 AI" 연출
        with st.status("AI가 분석을 시작합니다...", expanded=True) as status:
            time.sleep(1)
            status.update(label="데이터 무결성 검증 중...")
            df_main = pd.read_csv(main_file, encoding='cp949', low_memory=False)
            time.sleep(2)
            status.update(label="500개 의료 코드 교차 분석 및 패턴 학습 중...")
            time.sleep(3)
            status.update(label="다차원 공간에서 이상 패턴 탐색 중...")
            results, fig, total_claims, total_anomalies = run_analysis(df_main, disease_file, drug_file)
            time.sleep(2)
            status.update(label="분석 완료! 대시보드를 생성합니다.", state="complete", expanded=False)
        
        st.success("🎉 분석이 완료되었습니다! 아래 대시보드에서 결과를 확인하세요.")
        st.markdown("---")

        # 분석 결과 대시보드
        col1, col2, col3 = st.columns(3)
        col1.metric("총 진료 건수", f"{total_claims:,} 건")
        col2.metric("탐지된 이상치", f"{total_anomalies:,} 건", f"{(total_anomalies/total_claims):.2%}")
        col3.metric("분석된 특성(항목) 수", "500 개")
        
        st.markdown("---")

        tab1, tab2 = st.tabs(["📊 이상치 요약 및 그래프", "📑 Top 20 상세 분석"])

        with tab1:
            st.header("이상치 분포 시각화")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("가장 의심스러운 진료 Top 20")
            
            for res in reversed(results):
                expander_title = f"**Rank {res['rank']}** | 환자번호: {res['patient_id']} | 진료일: {res['date']}"
                with st.expander(expander_title):
                    st.write("▶ **이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):**")
                    st.dataframe(res['reasons'])
                    
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.exception(e)
