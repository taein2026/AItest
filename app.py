# app.py (동적 데이터 분석 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v11.0",
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
                Your persona is an **Elite AI Medical Analyst**. Your tone is direct, professional, and provides actionable intelligence.
                1.  **DYNAMIC ANALYSIS:** You MUST dynamically extract and cite specific item names and percentages from the `Input Data` table in your analysis. Do not use generic examples; use the real data provided for this specific patient.
                2.  **NO HALLUCINATION:** Base ALL statements *strictly* on the `Input Data`. Your role is to state what the data shows and what requires human review. Do not invent clinical facts.
                3.  **NO BOXES OR WEAK LANGUAGE:** Use bolding (`**text**`) for emphasis. State your findings as direct, professional conclusions.

                **Input Data:**
                - **Total Claims Analyzed:** {total_claims:,}
                - **Anomalous Patterns Detected:** {total_anomalies:,}
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**
                - **Data for Patient **{most_common_patient_id}** (This is the core evidence you must analyze):**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework (Follow this structure precisely):**

                ---

                ### **🔬 MediCopilot AI: 심층 분석 및 실행 권고 보고서**

                #### **분석 요약: 즉각적인 검토가 필요한 사례 발견**
                총 **{total_claims:,}**건의 진료 기록 분석 결과, 통계적으로 매우 이례적인 패턴 **{total_anomalies:,}**건을 식별했습니다. 특히 환자 ID **{most_common_patient_id}**의 사례는 명백한 통계적 특이점을 보여, 즉각적인 다각도 검토가 필요합니다.

                ---

                #### **분야별 전문가 검토 의견**

                ##### **1. 임상의학 관점 (Clinical Peer-Review Perspective)**
                **의견:** `Input Data`의 상세 보고서에 따르면, 환자 **{most_common_patient_id}**에게는 **(여기서 Input Data 테이블의 '특성명 (한글)' 컬럼에서 주요 항목 2-3개를 직접 언급하세요)** 등이 동시에 처방 및 진단되었습니다. 이처럼 통계적으로 매우 드문 조합은, 해당 환자의 복합적인 상태를 고려한 깊은 의학적 판단일 수도 있으나, 동시에 약물 상호작용의 위험성 등 의학적 타당성에 대한 **동료 전문가의 심도 있는 검토(Peer Review)가 반드시 필요**한 사례입니다.

                ##### **2. 보건 행정 및 심사 관점 (Audit & Regulatory Perspective)**
                **의견:** `Input Data`에서 확인된 처방 조합은 통계적 희귀성으로 인해 건강보험심사평가원의 **정밀 심사 또는 삭감 대상으로 지정될 명백한 위험**을 내포하고 있습니다. `Input Data`에 명시된 낮은 '평균 사용률'은 심사 과정에서 주된 검토 사유가 될 것입니다. 따라서 잠재적인 행정적 불이익을 방지하기 위해, 해당 처방의 **필요성을 입증할 수 있는 객관적인 의무기록과 상세한 소견서 등의 준비가 시급**합니다.

                ##### **3. 데이터 분석 관점 (Data Science Perspective)**
                **의견:** 이 패턴이 '이상치'로 탐지된 이유는 명확합니다. `Input Data`에 따르면, 환자 **{most_common_patient_id}**에게 적용된 처방들은 각각의 '평균 사용률(%)'이 매우 낮습니다. **(여기서 Input Data 테이블의 '특성명 (한글)'과 '평균 사용률 (%)'을 2-3개 직접 인용하여 구체적인 수치를 제시하세요. 예: "'A약물'의 사용률은 0.15%, 'B진단'의 사용률은 0.50%에 불과합니다.")** 통계적으로, 이러한 낮은 확률의 이벤트들이 한 개인에게 동시에 발생할 확률은 극히 희박합니다. 이 강력한 통계적 신호는 다음 두 가지 가능성을 지목합니다.
                * **가설 1 (임상적 특이성):** 환자가 매우 복합적이고 희귀한 상태에 있어, 불가피하게 이례적인 처방이 이루어졌을 가능성.
                * **가설 2 (데이터 입력 오류):** 진료비 청구 코드 입력 과정에서 실수가 발생했을 가능성. 통계적으로는 이 가설 또한 충분히 검토할 가치가 있습니다.

                ---

                #### **최종 권고: 즉시 실행이 필요한 조치**
                1.  **사실 관계 확정:** 환자 **{most_common_patient_id}**의 원본 의무기록과 청구 데이터를 대조하여, 데이터 입력 오류 여부를 최우선으로 검증하십시오.
                2.  **의학적 근거 확보:** 데이터에 오류가 없다면, 해당 처방의 타당성을 입증할 수 있는 상세한 임상적 소견 및 관련 근거 자료를 즉시 확보하고 문서화하십시오.
                3.  **동료 검토 시행:** 원내 동료 전문가 또는 관련 학회에 본 사례에 대한 자문을 구해, 처방의 적정성에 대한 객관적인 의견을 확보할 것을 권고합니다.

                **본 분석의 한계:** 본 AI 보고서는 통계적 패턴에 기반한 의사결정 지원 도구이며, 최종적인 의학적 판단을 대체할 수 없습니다. 모든 권고 사항의 최종 실행 여부는 해당 분야 전문가의 책임 하에 결정되어야 합니다.
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
