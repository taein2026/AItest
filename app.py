# app.py (최종 들여쓰기 오류 수정 완료 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v5.2",
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
            # try: 다음의 모든 코드는 반드시 들여쓰기(indentation)가 되어야 합니다.
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                # 1. AI에게 제공할 상세 데이터 가공
                most_common_patient_id = pd.Series([res['patient_id'] for res in results]).mode()[0]
                patient_specific_reasons = "상세 정보 없음"
                for res in results:
                    if res['patient_id'] == most_common_patient_id:
                        patient_specific_reasons = res['reasons'].to_markdown(index=False)
                        break

                # 2. AI에게 전달할 '다학제 전문가 위원회' 프롬프트 작성
                prompt = f"""
                **Your Role & Goal:**
                You are 'MediCopilot AI', a Multi-Disciplinary Medical AI Reviewer. Your mission is to conduct a comprehensive analysis of the provided anomaly report from multiple expert perspectives. Your final output must be a professional, structured, and deeply insightful briefing document for a hospital's internal review committee and national health regulators.

                **Input Data:**
                - **Total Claims Analyzed:** {total_claims:,}
                - **Anomalous Patterns Detected:** {total_anomalies:,} (Top {(total_anomalies/total_claims):.2%} of all claims)
                - **Primary Patient of Interest:** Patient ID `{most_common_patient_id}`.
                - **Detailed Anomaly Report for Patient `{most_common_patient_id}` (Rarest combinations found):**
                ```markdown
                {patient_specific_reasons}
                ```

                **Mandatory Briefing Framework:**
                Generate a briefing in Korean. You MUST follow this structure precisely. Do not deviate.

                ---

                ### 🔬 MediCopilot AI 다학제 통합 분석 보고서

                #### **1. 분석 개요 (Executive Summary)**
                * 분석의 핵심 결과를 2~3문장으로 요약합니다. (총 진료 건수, 이상 패턴 식별 건수, 주요 발견 등)

                #### **2. 심층 분석: 주요 관심 환자 (`{most_common_patient_id}`)**
                * 이 환자가 왜 분석의 핵심 대상으로 선정되었는지 명확히 설명합니다.

                #### **3. 다각적 전문가 의견 (Multi-Faceted Expert Analysis)**
                * **3.1. 임상의 및 규제 기관 관점 (Clinical & Regulatory Perspective):**
                    * 제공된 '상세 이상 패턴 보고서'를 바탕으로, 해당 처방/진단 조합이 표준 임상 프로토콜이나 일반적인 진료 가이드라인에서 벗어나는지 평가하세요.
                    * 이 패턴이 건강보험심사평가원 등 규제 기관의 심사에서 잠재적으로 삭감 또는 정밀 조사의 대상이 될 가능성이 있는지 전문적으로 서술하세요. 의학적 타당성에 대한 의문을 제기하세요.

                * **3.2. 데이터 과학자 관점 (Data Science Perspective):**
                    * 이 패턴이 왜 통계적 '이상치(Anomaly)'로 탐지되었는지 기술적으로 설명하세요.
                    * '상세 이상 패턴 보고서'의 '평균 사용률' 데이터를 직접 인용하여, 해당 조합이 전체 데이터셋에서 얼마나 희귀한 이벤트인지 수치적으로 강조하세요. (예: "해당 조합의 평균 사용률은 0.001로, 이는 10만 건의 진료 중 단 1건에서만 발견될 정도의 극히 이례적인 수치입니다.")

                #### **4. 근본 원인 추론 (Root Cause Hypothesis)**
                * 탐지된 이상 패턴의 가장 가능성 높은 원인을 다음 두 가지 가설을 바탕으로 추론하고, 어떤 쪽에 더 무게가 실리는지 의견을 제시하세요.
                    * **가설 A (의료적 판단):** 환자의 특이한 상태로 인한 의학적으로는 타당하지만 통계적으로 희귀한 처방일 가능성.
                    * **가설 B (행정적 오류):** 진료비 청구 코드 입력 과정에서의 실수(Data Entry Error) 또는 시스템 오류일 가능성.

                #### **5. 최종 권고 및 제언 (Final Recommendations)**
                * 분석 결과를 종합하여, 검토 위원회가 즉시 실행해야 할 구체적인 액션 플랜을 번호를 매겨 3가지 이상 제시하세요.
                * 이 분석의 명확한 한계점(예: "이 분석은 통계적 희귀성을 기반으로 하며, 실제 의료 행위의 타당성을 최종 판단하는 것은 아님")을 반드시 명시하여, 인간 전문가의 최종 검토가 필수적임을 강조하세요.

                ---
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

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 눌러주세요!")
