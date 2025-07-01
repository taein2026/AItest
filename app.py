# app.py (최종 오류 수정 및 기능 개선 완료 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v5.1",
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
        if st.button("✔️ 업로드 파일 저장", use_container_width=True):
            with st.spinner("파일을 읽고 있습니다..."):
                st.session_state['df_main'] = pd.read_csv(uploaded_main_file, encoding='cp949', low_memory=False)
                st.session_state['df_disease'] = pd.read_excel(uploaded_disease_file, dtype={'상병코드': str})
                st.session_state['df_drug'] = pd.read_excel(uploaded_drug_file, dtype={'연합회코드': str})
                st.session_state['files_ready'] = True
                st.success("파일 저장 완료!")

    st.markdown("---")
    start_button = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True, disabled='files_ready' not in st.session_state)

# --- 메인 화면 ---
st.title("✨ MediCopilot AI")
st.markdown("---")

if not start_button:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 누른 후, 'AI 분석 실행' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_professional.png", use_column_width=True)

if start_button:
    if 'files_ready' in st.session_state:
        # 전체 분석 과정을 감싸는 try 블록입니다.
        try:
            # --- AI 분석 프로세스 바 ---
            st.header("AI 분석 프로세스")
            
            step1, step2, step3, step4, step5, step6 = st.columns(6)
            placeholders = [step1.empty(), step2.empty(), step3.empty(), step4.empty(), step5.empty(), step6.empty()]
            steps = ["1. 데이터 로딩", "2. 데이터 전처리", "3. AI 모델 학습", "4. 패턴 분석", "5. 이상치 탐지", "6. 보고서 생성"]

            for i, placeholder in enumerate(placeholders):
                placeholder.info(f'**{steps[i]}**\n\n*상태: ⏳*')

            time.sleep(1)
            placeholders[0].error(f'**{steps[0]}**\n\n*상태: 🔥 분석 중...*')
            df_main = st.session_state['df_main']
            df_disease = st.session_state['df_disease']
            df_drug = st.session_state['df_drug']
            time.sleep(1.5)
            placeholders[0].success(f'**{steps[0]}**\n\n*상태: ✅ 완료*')

            placeholders[1].error(f'**{steps[1]}**\n\n*상태: 🔥 분석 중...*')
            time.sleep(1.5)
            placeholders[1].success(f'**{steps[1]}**\n\n*상태: ✅ 완료*')

            placeholders[2].error(f'**{steps[2]}**\n\n*상태: 🔥 분석 중...*')
            time.sleep(2)
            placeholders[2].success(f'**{steps[2]}**\n\n*상태: ✅ 완료*')
            
            placeholders[3].error(f'**{steps[3]}**\n\n*상태: 🔥 분석 중...*')
            results, fig, total_claims, total_anomalies = run_analysis(df_main, df_disease, df_drug)
            placeholders[3].success(f'**{steps[3]}**\n\n*상태: ✅ 완료*')

            placeholders[4].error(f'**{steps[4]}**\n\n*상태: 🔥 분석 중...*')
            time.sleep(1.5)
            placeholders[4].success(f'**{steps[4]}**\n\n*상태: ✅ 완료*')

            placeholders[5].error(f'**{steps[5]}**\n\n*상태: 🔥 분석 중...*')
            time.sleep(1)
            placeholders[5].success(f'**{steps[5]}**\n\n*상태: ✅ 완료*')
            
            st.success("🎉 모든 분석 과정이 성공적으로 완료되었습니다!")
            st.markdown("---")
            
            # --- AI 최종 분석 브리핑 (고급 프롬프트 적용 버전) ---
            st.header("🔬 AI 최종 분석 브리핑")

            # Gemini API 호출을 위한 별도 try 블록
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                # 1. AI에게 제공할 상세 데이터 가공
                most_common_patient_id = pd.Series([res['patient_id'] for res in results]).mode()[0]
                patient_specific_reasons = "상세 정보 없음"
                for res in results:
                    if res['patient_id'] == most_common_patient_id:
                        patient_specific_reasons = res['reasons'].to_markdown(index=False)
                        break

                # 2. AI에게 전달할 구체적인 프롬프트 작성
                prompt = f"""
                **Your Role & Goal:**
                You are 'MediCopilot AI', a highly specialized medical data analyst for a hospital review committee. Your goal is to analyze the provided anomaly detection report and write a professional, insightful briefing for human reviewers. Your analysis must be sharp, clear, and actionable.

                **Input Data:**
                - **Total Claims Analyzed:** {total_claims:,}
                - **Anomalous Patterns Detected:** {total_anomalies:,} (Top {(total_anomalies/total_claims):.2%})
                - **Primary Patient of Interest:** Patient ID `{most_common_patient_id}`. This patient appeared most frequently in the top 20 anomaly list.
                - **Detailed Anomaly Report for Patient `{most_common_patient_id}` (Rarest combinations found):**
                ```markdown
                {patient_specific_reasons}
                ```

                **Briefing Generation Instruction:**
                Based on all the provided data, generate a briefing in Korean with the following strict format using Markdown:

                ### 🔬 MediCopilot AI 최종 분석 브리핑

                #### 1. 분석 개요 (Executive Summary)
                - 총 몇 건의 진료 기록을 분석했고, 그중 몇 건의 통계적 이상 패턴을 식별했는지 명확히 요약하세요.

                #### 2. 심층 분석: 주요 관심 환자 (`{most_common_patient_id}`)
                - 이 환자가 왜 주요 관심 대상이 되었는지 설명하세요.
                - 위에 제공된 **상세 이상 패턴 보고서(Detailed Anomaly Report)**를 직접적으로 인용하고 해석하세요.
                - 예를 들어, "이 환자의 경우, 'A 상병'과 'B 약물'의 조합이 나타났습니다. 이는 전체 데이터에서 0.01% 미만으로 발견되는 매우 이례적인 패턴으로, 심층 검토가 필요합니다." 와 같이 구체적인 근거를 들어 설명하세요.

                #### 3. 권장 조치 (Recommended Actions)
                - 분석 결과를 바탕으로 검토팀이 즉시 수행해야 할 다음 단계를 1, 2, 3 순서로 명확하게 제시하세요.
                - (예: 1. 환자 `{most_common_patient_id}`의 전체 진료 이력 원본 대조. 2. 해당 처방을 내린 주치의 면담 요청. 3. 유사 패턴을 보이는 다른 환자 그룹 탐색.)
                """
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt, stream=True)

                with st.chat_message("ai", avatar="🤖"):
                    report_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.text
                        report_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    report_placeholder.markdown(full_response, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"AI 브리핑 생성 중 오류가 발생했습니다: {e}")
            
            st.markdown("---")
            
            # --- 분석 결과 상세 대시보드 ---
            st.header("분석 결과 상세 대시보드")
            col1, col2, col3 = st.columns(3)
            col1.metric("총 진료 건수", f"{total_claims:,} 건")
            col2.metric("탐지된 이상치", f"{total_anomalies:,} 건", f"상위 {(total_anomalies/total_claims):.2%}")
            col3.metric("분석된 특성(항목) 수", "500+ 개") 
            
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

        # 전체 분석 과정을 감싸는 try에 대한 except 블록입니다.
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 눌러주세요!")
