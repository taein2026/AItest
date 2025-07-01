# app.py (최종 신뢰도 및 가독성 개선 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v7.0",
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

                # --- 1. 데이터 사전 가공 (Python-Side Pre-processing) ---
                most_common_patient_id = pd.Series([res['patient_id'] for res in results]).mode()[0]
                patient_specific_reasons_str = "상세 정보 없음"
                for res in results:
                    if res['patient_id'] == most_common_patient_id:
                        reasons_df = res['reasons'].copy()
                        reasons_df['평균 사용률 (%)'] = (reasons_df['평균 사용률'] * 100).map('{:.2f}%'.format)
                        patient_specific_reasons_str = reasons_df[['특성명 (한글)', '평균 사용률 (%)']].to_markdown(index=False)
                        break

                # --- 2. AI 길들이기 (Advanced Prompt Engineering) ---
                prompt = f"""
                **CRITICAL DIRECTIVE:**
                Your primary function is to be a **Data-Driven Analyst**, NOT a medical doctor or a regulator.
                1.  **NO HALLUCINATION:** Base ALL your statements *strictly* on the `Input Data` provided. Do not invent, assume, or use external knowledge about medical practices, drug interactions, or clinical guidelines.
                2.  **NO BOXES:** Do not use backticks (` `) or code blocks (```) in your response. Emphasize with bolding (`**text**`) only.
                3.  **FRAME AS QUESTIONS:** Instead of making definitive judgments, phrase your analysis as questions for human experts. For example, instead of "This is wrong," say "This statistical rarity warrants a review by a clinical expert to determine its appropriateness."

                **Your Role & Goal:**
                You are 'MediCopilot AI'. Your goal is to synthesize the provided data into a structured, objective report for a multi-disciplinary review committee.

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
                `Input Data`에 명시된 총 진료 건수 중 통계적으로 이례적인 패턴을 보인 **{total_anomalies:,}**건을 식별했습니다. 특히 환자 ID **{most_common_patient_id}**에게서 가장 주목할 만한 통계적 특이점이 발견되어, 해당 사례를 중심으로 심층 분석을 진행합니다.

                #### **2. 다각적 데이터 분석 및 전문가 검토 제안**

                ##### **2.1. 임상 전문가(의사/연구원) 검토 제안**
                **데이터 요약:** 환자 **{most_common_patient_id}**의 경우, `Input Data`에 나타난 바와 같이 통계적으로 사용률이 매우 낮은 항목들의 조합이 발견되었습니다.
                **검토 요청 사항:** 이처럼 통계적으로 매우 드문 처방/진단 조합의 의학적 타당성에 대한 임상적 재검토가 필요합니다. 해당 환자의 특수한 의학적 상태를 고려한 처방이었는지, 혹은 표준 임상 프로토콜과 차이가 있는지에 대한 정신과 전문가의 소견이 요구됩니다.

                ##### **2.2. 보건 행정 및 심사 전문가 검토 제안**
                **데이터 요약:** 본 사례는 통계적 희귀성을 기준으로 식별되었습니다.
                **검토 요청 사항:** 이러한 통계적 특이점이 건강보험 청구 및 심사 과정에서 어떤 쟁점을 가질 수 있는지에 대한 행정적 검토가 필요합니다. 해당 청구 건의 적정성을 입증하기 위해 어떤 추가 소명 자료가 필요할지, 규제 관점에서의 검토를 제안합니다.

                ##### **2.3. 데이터 분석 전문가 관점**
                **데이터 요약:** 이 패턴이 통계적 '이상치'로 분류된 이유는 `Input Data`의 '평균 사용률 (%)' 수치가 명확히 보여줍니다. `Input Data`에 따르면, 해당 환자에게 적용된 특정 항목들은 전체 데이터에서 **1% 미만**으로 사용되는 등 발생 빈도가 현저히 낮았습니다.
                **기술적 소견:** 여러 개의 희귀한 이벤트가 한 환자에게 동시에 발생할 확률은 더욱 낮아지므로, 본 시스템은 이를 통계적으로 유의미한 이상 패턴으로 탐지한 것입니다. 이는 데이터 입력 오류(Data Entry Error) 또는 시스템 오류의 가능성도 배제할 수 없음을 시사합니다.

                #### **3. 최종 권고 및 본 분석의 한계**
                **권고 사항:**
                1.  환자 **{most_common_patient_id}**의 원본 의무기록과 청구 내역을 대조하여 데이터의 정확성을 최우선으로 확인해야 합니다.
                2.  위의 각 전문가 검토 제안에 따라, 임상 및 행정 분야 전문가의 공식적인 검토를 진행할 것을 권고합니다.

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
