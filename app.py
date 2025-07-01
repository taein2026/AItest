# app.py (전문가 톤 최종 수정 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v8.0",
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
            time.sleep(0.5)
            placeholders[0].success(f'**{steps[0]}**\n\n*상태: ✅ 완료*')
            placeholders[1].success(f'**{steps[1]}**\n\n*상태: ✅ 완료*')
            
            # --- 실제 분석 수행 ---
            df_main = st.session_state['df_main']
            df_disease = st.session_state['df_disease']
            df_drug = st.session_state['df_drug']
            results, fig, total_claims, total_anomalies = run_analysis(df_main, df_disease, df_drug)
            
            placeholders[2].success(f'**{steps[2]}**\n\n*상태: ✅ 완료*')
            placeholders[3].success(f'**{steps[3]}**\n\n*상태: ✅ 완료*')
            placeholders[4].success(f'**{steps[4]}**\n\n*상태: ✅ 완료*')
            time.sleep(0.5)
            placeholders[5].success(f'**{steps[5]}**\n\n*상태: ✅ 완료*')
            
            st.success("🎉 모든 분석 과정이 성공적으로 완료되었습니다!")
            st.markdown("---")
            
            # --- AI 최종 분석 브리핑 ---
            st.header("🔬 AI 최종 분석 브리핑")

            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                # --- 1. 데이터 사전 가공 ---
                most_common_patient_id = pd.Series([res['patient_id'] for res in results]).mode()[0]
                patient_specific_reasons_str = "상세 정보 없음"
                for res in results:
                    if res['patient_id'] == most_common_patient_id:
                        reasons_df = res['reasons'].copy()
                        reasons_df['평균 사용률 (%)'] = (reasons_df['평균 사용률'] * 100).map('{:.2f}%'.format)
                        patient_specific_reasons_str = reasons_df[['특성명 (한글)', '평균 사용률 (%)']].to_markdown(index=False)
                        break

                # --- 2. 최종 프롬프트 ---
                prompt = f"""
                **CRITICAL DIRECTIVE:**
                Your persona is a **confident, data-driven professional analyst**. Your tone must be assertive and declarative, not inquisitive.
                1.  **NO HALLUCINATION:** Base ALL statements *strictly* on the `Input Data`. Your role is to state what the data shows and what requires human review. Do not invent clinical facts.
                2.  **NO BOXES:** Do not use backticks (` `) or code blocks (``` ```) for emphasis. Use bolding (`**text**`) only.
                3.  **STATE CONCLUSIONS, DON'T ASK QUESTIONS:** Instead of asking "How should we review this?", you must state "This requires review." Present your findings as actionable conclusions for the committee.

                **Input Data:**
                - **Total Claims Analyzed:** {total_claims:,}
                - **Anomalous Patterns Detected:** {total_anomalies:,}
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**
                - **Data for Patient **{most_common_patient_id}**:**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework (Follow this structure precisely):**

                ---

                ### **🔬 MediCopilot AI 다학제 통합 분석 보고서**

                #### **1. 분석 개요**
                총 **{total_claims:,}**건의 진료 기록 분석 결과, 통계적으로 유의미한 이상 패턴 **{total_anomalies:,}**건을 식별했습니다. 특히 환자 ID **{most_common_patient_id}**의 사례에서 가장 주목할 만한 통계적 특이점이 발견되어, 해당 사례를 중심으로 심층 분석을 제시합니다.

                #### **2. 전문가 관점별 데이터 분석 및 검토 의견**

                ##### **2.1. 임상 전문가 (의사/연구원) 관점**
                **분석 의견:** 환자 **{most_common_patient_id}**의 사례는 `Input Data`에서 확인된 바와 같이, 통계적으로 매우 드물게 조합되는 처방 및 진단이 동시에 이루어졌습니다. 이 이례적인 패턴은 해당 처방의 의학적 타당성과 환자의 특수성에 대한 **반드시 필요한 임상적 재검토를 요구합니다.** 검토 시, 해당 처방 조합이 표준 임상 프로토콜에 부합하는지 여부를 중점적으로 확인해야 합니다.

                ##### **2.2. 보건 행정 및 심사 전문가 관점**
                **분석 의견:** 본 사례처럼 통계적 희귀성이 높은 청구 건은 건강보험심사평가원 등 규제 기관의 정밀 심사 또는 현지 조사 대상으로 선정될 가능성이 있습니다. 따라서, 해당 처방의 **의학적 근거와 사유를 명확히 소명할 수 있는 객관적인 기록 확보가 필수적입니다.** 이는 잠재적인 행정적 불이익을 방지하기 위한 선제적 조치입니다.

                ##### **2.3. 데이터 분석 전문가 관점**
                **분석 의견:** 기술적으로 이 패턴은 명백한 '통계적 이상치'입니다. `Input Data`의 '평균 사용률 (%)' 수치는 각 항목이 개별적으로도 희귀함을 보여주며, 이러한 희귀 항목들의 동시 발생 확률은 훨씬 낮습니다. 이 강력한 통계적 신호는 **환자의 임상적 특이성 또는 데이터 입력 오류(Data Entry Error)라는 두 가지 가능성을 모두 시사**하므로, 두 가설에 대한 검증이 모두 필요합니다.

                #### **3. 최종 권고 및 본 분석의 한계**
                **권고 사항:**
                1.  **데이터 정확성 검증:** 환자 **{most_common_patient_id}**의 원본 의무기록과 청구 데이터를 대조하여 사실 관계를 최우선으로 확정해야 합니다.
                2.  **전문가 공식 검토:** 위 분석 의견에 근거하여, 임상 및 행정 전문가로 구성된 위원회의 공식적인 검토를 진행할 것을 강력히 권고합니다.

                **명확한 한계 고지:**
                **본 AI 보고서는 통계적 패턴 분석을 통해 인간 전문가의 검토가 필요한 대상을 식별하는 보조 도구입니다. 이 보고서는 의료 행위의 적정성을 최종 판단하지 않으며, 모든 해석과 결정은 반드시 해당 분야의 인간 전문가에 의해 이루어져야 합니다.**

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
            
            # --- 대시보드 및 상세 분석 테이블 ---
            st.markdown("---")
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
                    expander_title = f"**Rank {res['rank']}** | 환자번호: **{res['patient_id']}** | 진료일: **{res['date']}**"
                    with st.expander(expander_title):
                        st.write("▶ **이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):**")
                        st.dataframe(res['reasons'], use_container_width=True) 

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 눌러주세요!")
