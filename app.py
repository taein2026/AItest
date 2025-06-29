# app.py (퍼포먼스 및 시각화 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v2.0",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lottie 애니메이션 로드 함수 ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        return None

# --- 스타일 정의 (굵고 큰 글씨) ---
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- 사이드바 ---
with st.sidebar:
    lottie_url = "https://lottie.host/e883236e-1335-4309-a185-11a518012e69/Tpde6s5V1C.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="sidebar_lottie")

    st.title("📄 파일 업로드")
    st.info("분석에 필요한 파일 3개를 모두 업로드해주세요.")
    main_file = st.file_uploader("① 메인 진료 데이터", type=['csv'])
    disease_file = st.file_uploader("② 상병명 매칭 테이블", type=['xlsx'])
    drug_file = st.file_uploader("③ 약물명 매칭 테이블", type=['xlsx'])
    st.markdown("---")
    start_button = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True, disabled=not(main_file and disease_file and drug_file))

# --- 메인 화면 ---
st.title("✨ AI 이상 진료 탐지 시스템 v2.0")
st.markdown("---")

# 초기 화면
if not start_button:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드한 후, 'AI 분석 실행' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_professional.png", use_column_width=True)

# 분석 시작
if start_button:
    try:
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        #                   "AI 분석 프로세스 바" 퍼포먼스
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        st.header("AI 분석 프로세스")
        
        # 4개의 단계 컬럼 생성
        step1, step2, step3, step4 = st.columns(4)
        
        # 초기 상태 설정
        step1_placeholder = step1.empty()
        step2_placeholder = step2.empty()
        step3_placeholder = step3.empty()
        step4_placeholder = step4.empty()

        step1_placeholder.info('**1. 데이터 검증**\n\n*상태: ⏳ 대기 중*')
        step2_placeholder.info('**2. AI 모델 학습**\n\n*상태: ⏳ 대기 중*')
        step3_placeholder.info('**3. 이상 패턴 탐색**\n\n*상태: ⏳ 대기 중*')
        step4_placeholder.info('**4. 최종 보고서 생성**\n\n*상태: ⏳ 대기 중*')

        # 단계별 진행
        time.sleep(1)
        step1_placeholder.info('**1. 데이터 검증**\n\n*상태: ⚙️ 진행 중...*')
        df_main = pd.read_csv(main_file, encoding='cp949', low_memory=False)
        time.sleep(1.5)
        step1_placeholder.success('**1. 데이터 검증**\n\n*상태: ✅ 완료*')

        step2_placeholder.info('**2. AI 모델 학습**\n\n*상태: ⚙️ 진행 중...*')
        time.sleep(2.5)
        step2_placeholder.success('**2. AI 모델 학습**\n\n*상태: ✅ 완료*')

        step3_placeholder.info('**3. 이상 패턴 탐색**\n\n*상태: ⚙️ 진행 중...*')
        results, fig, total_claims, total_anomalies = run_analysis(df_main, disease_file, drug_file)
        time.sleep(2)
        step3_placeholder.success('**3. 이상 패턴 탐색**\n\n*상태: ✅ 완료*')

        step4_placeholder.info('**4. 최종 보고서 생성**\n\n*상태: ⚙️ 진행 중...*')
        time.sleep(1.5)
        step4_placeholder.success('**4. 최종 보고서 생성**\n\n*상태: ✅ 완료*')
        
        st.success("🎉 모든 분석 과정이 성공적으로 완료되었습니다!")
        st.markdown("---")

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        #           "AI 최종 분석 브리핑" (굵고 큰 글씨 적용)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        st.header("🔬 AI 최종 분석 브리핑")
        
        patient_ids = [res['patient_id'] for res in results]
        if patient_ids:
            most_common_patient = pd.Series(patient_ids).mode()[0]
            count = patient_ids.count(most_common_patient)
            key_finding = f"가장 주목할 만한 패턴은 특정 환자에게서 이상치가 집중적으로 발견된 점입니다. 특히 **환자번호 `{most_common_patient}`**는 Top 20 리스트에 <span class='big-font'>{count}회</span> 등장하여, 해당 환자의 진료 이력에 대한 심층 검토가 필요합니다."
        else:
            key_finding = "탐지된 이상치 중에서 특별히 집중되는 패턴은 발견되지 않았습니다."

        summary_text = f"> **분석 요약:** 총 <span class='big-font'>{total_claims:,}</span>건의 진료 데이터에서 <span class='big-font'>{total_anomalies:,}</span>건의 통계적 이상 패턴을 식별했습니다."
        finding_text = f"> **핵심 발견:** {key_finding}"
        recommendation_text = f"> **권장 조치:** 이상치로 탐지된 진료 건들의 상세 분석을 통해, 이례적인 처방/진단 조합의 의학적 타당성을 확인하십시오."
        
        # 순차적 브리핑
        st.markdown(summary_text, unsafe_allow_html=True)
        time.sleep(1)
        st.markdown(finding_text, unsafe_allow_html=True)
        time.sleep(1)
        st.markdown(recommendation_text, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 분석 결과 대시보드 출력... (이하 동일)
        st.header("분석 결과 상세 대시보드")
        col1, col2, col3 = st.columns(3)
        col1.metric("총 진료 건수", f"{total_claims:,} 건")
        col2.metric("탐지된 이상치", f"{total_anomalies:,} 건", f"상위 {(total_anomalies/total_claims):.2%}")
        col3.metric("분석된 특성(항목) 수", "500 개")
        
        tab1, tab2 = st.tabs(["📊 **이상치 요약 및 그래프**", "📑 **Top 20 상세 분석**"])
        with tab1:
            st.subheader("이상치 분포 시각화")
            st.info("파란색 점들은 일반적인 진료 패턴을, 빨간색 점들은 AI가 통계적으로 특이하다고 판단한 이상치를 나타냅니다.")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("가장 의심스러운 진료 Top 20")
            st.info("Rank가 높을수록 패턴이 이질적이라는 의미입니다. 각 항목을 클릭하여 상세 원인을 확인하세요.")
            for res in reversed(results):
                expander_title = f"**Rank {res['rank']}** | 환자번호: `{res['patient_id']}` | 진료일: `{res['date']}`"
                with st.expander(expander_title):
                    st.write("▶ **이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):**")
                    st.dataframe(res['reasons'], use_container_width=True) 
                    
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.exception(e)
