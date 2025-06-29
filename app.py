# app.py (타이핑 효과 복구 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v3.0",
    page_icon="🧠",
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

# --- 사이드바 ---
with st.sidebar:
    lottie_url = "https://lottie.host/e883236e-1335-4309-a185-11a518012e69/Tpde6s5V1C.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="sidebar_lottie")

    st.title("📄 파일 업로드")
    st.info("분석에 필요한 파일 3개를 모두 업로드해주세요.")
    
    uploaded_main_file = st.file_uploader("① 메인 진료 데이터", type=['csv'])
    uploaded_disease_file = st.file_uploader("② 상병명 매칭 테이블", type=['xlsx'])
    uploaded_drug_file = st.file_uploader("③ 약물명 매칭 테이블", type=['xlsx'])
    
    if uploaded_main_file and uploaded_disease_file and uploaded_drug_file:
        st.session_state['files_ready'] = True

    st.markdown("---")
    start_button = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True, disabled='files_ready' not in st.session_state)

# --- 메인 화면 ---
st.title("🧠 AI 이상 진료 탐지 시스템 v3.0")
st.markdown("---")

# 초기 화면
if not start_button:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드하고, 'AI 분석 실행' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_professional.png", use_column_width=True)

# 분석 시작
if start_button:
    if 'files_ready' in st.session_state:
        try:
            # --- AI 분석 프로세스 바 ---
            st.header("AI 분석 프로세스")
            step1, step2, step3, step4 = st.columns(4)
            s1_placeholder, s2_placeholder, s3_placeholder, s4_placeholder = step1.empty(), step2.empty(), step3.empty(), step4.empty()
            placeholders = [s1_placeholder, s2_placeholder, s3_placeholder, s4_placeholder]
            steps = ["1. 데이터 로딩", "2. 데이터 학습", "3. 이상치 탐지", "4. 보고서 생성"]

            for i, placeholder in enumerate(placeholders):
                placeholder.info(f'**{steps[i]}**\n\n*상태: ⏳ 대기 중*')

            time.sleep(1)
            s1_placeholder.info(f'**{steps[0]}**\n\n*상태: ⚙️ 진행 중...*')
            df_main = pd.read_csv(uploaded_main_file, encoding='cp949', low_memory=False)
            df_disease = pd.read_excel(uploaded_disease_file, dtype={'상병코드': str})
            df_drug = pd.read_excel(uploaded_drug_file, dtype={'연합회코드': str})
            time.sleep(1.5)
            s1_placeholder.success(f'**{steps[0]}**\n\n*상태: ✅ 완료*')

            s2_placeholder.info(f'**{steps[1]}**\n\n*상태: ⚙️ 진행 중...*')
            results, fig, total_claims, total_anomalies = run_analysis(df_main, df_disease, df_drug)
            s2_placeholder.success(f'**{steps[1]}**\n\n*상태: ✅ 완료*')

            s3_placeholder.info(f'**{steps[2]}**\n\n*상태: ⚙️ 진행 중...*')
            time.sleep(2)
            s3_placeholder.success(f'**{steps[2]}**\n\n*상태: ✅ 완료*')

            s4_placeholder.info(f'**{steps[3]}**\n\n*상태: ⚙️ 진행 중...*')
            time.sleep(1.5)
            s4_placeholder.success(f'**{steps[3]}**\n\n*상태: ✅ 완료*')
            
            st.success("🎉 모든 분석 과정이 성공적으로 완료되었습니다!")
            st.markdown("---")

            # --- AI 최종 분석 브리핑 (타이핑 효과 적용) ---
            st.header("🔬 AI 최종 분석 브리핑")
            
            def stream_text(text_to_stream):
                placeholder = st.empty()
                full_response = ""
                for chunk in text_to_stream:
                    full_response += chunk
                    time.sleep(0.015)
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                placeholder.markdown(full_response, unsafe_allow_html=True)

            patient_ids = [res['patient_id'] for res in results]
            if patient_ids:
                most_common_patient = pd.Series(patient_ids).mode()[0]
                count = patient_ids.count(most_common_patient)
                key_finding = f"가장 주목할 만한 패턴은 특정 환자에게서 이상치가 집중적으로 발견된 점입니다. 특히 **환자번호 `{most_common_patient}`**는 Top 20 리스트에 <span style='color: #00f4d4;'>**{count}회**</span> 등장하여, 해당 환자의 진료 이력에 대한 심층 검토가 필요합니다."
            else:
                key_finding = "탐지된 이상치 중에서 특별히 집중되는 패턴은 발견되지 않았습니다."
            
            summary_text = f"> **분석 요약:** 총 <span style='color: #00f4d4;'>**{total_claims:,}**</span>건의 진료 기록을 분석하여, <span style='color: #00f4d4;'>**{total_anomalies:,}**</span>건의 통계적 이상 패턴을 식별했습니다."
            finding_text = f"> **핵심 발견:** {key_finding}"
            recommendation_text = f"> **권장 조치:** 이상치로 탐지된 진료 건들의 상세 분석을 통해, 이례적인 처방/진단 조합의 의학적 타당성을 확인하십시오."
            
            full_briefing_text = f"{summary_text}\n\n{finding_text}\n\n{recommendation_text}"
            stream_text(full_briefing_text)
            
            st.markdown("---")
            
            # --- 분석 결과 상세 대시보드 ---
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
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드해주세요!")
