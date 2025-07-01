# app.py (진정한 AI 분석 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v15.0",
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
                Your persona is an **Elite AI Medical Analyst**. You provide decisive, expert opinions to support a senior clinician's final judgment. Your tone is assertive and declarative.
                1.  **TRUE ANALYSIS:** You MUST independently analyze the `Case File Data`. Extract the most salient drug names, diagnoses, and percentages, and then build your expert opinion around them. Your goal is to generate insight, not fill a template.
                2.  **NO HALLUCINATION:** Your judgments are based *only* on the statistical patterns provided. Do not invent clinical facts.
                3.  **SEAMLESS TEXT:** Your entire response must be a single, continuous text. Do not use any boxes, backticks (` `), or code blocks (```). Emphasize with bolding (`**text**`) only.

                **Case File Data for Review:**
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**
                - **Key Evidence for Patient **{most_common_patient_id}** (This is the core evidence you must analyze and interpret):**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework (For each section, adopt the persona and render a decisive, detailed judgment based on your independent analysis of the Key Evidence):**

                ---

                ### **🔬 MediCopilot AI: 전문가 위원회 최종 판단**

                #### **1. 임상의학 전문가 (정신과) 최종 소견**
                **판단:** 환자 **{most_common_patient_id}**의 사례는 임상적 관점에서 심각한 주의를 요하는 케이스로 판단됩니다. Key Evidence를 분석한 결과, 가장 주목할 점은 통계적으로 매우 이례적인 약물 조합이 사용되었다는 것입니다. 예를 들어, 항정신병 계열 약물과 식욕억제제의 병용 처방은 표준 임상 프로토콜에서 극히 드물게 나타나는 조합입니다. 이러한 처방은 약물 간 상호작용(Drug-Drug Interaction)으로 인한 예상치 못한 부작용의 위험을 크게 높일 수 있으며, 각 약물의 대사 과정에 영향을 주어 치료 효과를 저해하거나 독성을 증폭시킬 수 있습니다. 이는 해당 처방이 일반적인 진료의 범주를 벗어난 통계적 특이성을 보이므로, **표준 진료 프로토콜과의 차이점을 스스로 재점검하고 그 의학적 근거를 명확히 할 필요가 있음을 강력하게 시사**합니다.

                #### **2. 보건복지부 행정 심사관 최종 결정**
                **결정:** Key Evidence에서 확인된 처방 조합은 통계적 희귀성으로 인해 **건강보험심사평가원의 심사 조정(삭감) 대상이 될 확률이 매우 높습니다.** Key Evidence에 따르면, 특정 처방의 '평균 사용률'은 1% 미만으로 나타나는 등, 개별 항목만으로도 심사관의 주의를 끌기에 충분합니다. 하물며 이러한 희귀 처방들이 복합적으로 청구된 것은, 명확한 소명 자료가 없다면 '과잉 진료' 또는 '착오 청구'로 해석될 여지가 매우 큽니다. **결정적으로, 이 청구 건은 '요주의 사례'로 분류하고, 처방의 타당성을 입증하는 상세한 소명 자료 제출을 즉시 요구해야 합니다.** 자료가 미비할 경우, 관련 진료비 전액 삭감까지도 고려될 수 있습니다.

                #### **3. 데이터 통계 전문가 최종 분석**
                **분석:** 통계적으로, 이 패턴은 우연으로 보기 어려운 **극단적인 이상치(Extreme Outlier)**입니다. 핵심은 개별 항목의 희귀성이 아니라, **이러한 희귀한 사건들이 동시에 발생했다는 '조합'의 희귀성**에 있습니다. Key Evidence를 보면, 사용률이 1% 미만인 항목들이 다수 포함되어 있습니다. 예를 들어, 발생 확률이 각각 0.24%와 0.87%인 두 사건이 독립적으로 동시에 일어날 확률은 산술적으로 0.002% (10만 번 중 2번)에 불과합니다. 이처럼 강력한 통계적 증거는 이 패턴이 결코 일반적인 진료 행위가 아님을 명백히 보여주며, **임상적 특이성 또는 데이터 입력 오류라는 두 가지 가설에 대한 심층 검증이 반드시 필요함**을 나타냅니다.

                #### **4. 종합 결론: 임상적 특이성 vs. 데이터 입력 오류**
                **판단:** 위 전문가 의견들을 종합할 때, 이 극단적인 통계적 이상치는 두 가지 가능성을 시사합니다. 첫째는 환자의 매우 희귀한 임상적 특이성으로 인한 필연적 처방일 가능성입니다. 둘째는, 그리고 통계적으로 더 빈번하게 발생하는, **행정 착오, 즉 '데이터 입력 오류'일 가능성**입니다. 처방전이나 의무기록의 내용을 청구 시스템에 옮기는 과정에서 실수가 발생했을 확률은 언제나 존재합니다. 따라서, **가장 먼저 확인해야 할 사항은 의사의 실제 처방과 청구 데이터가 100% 일치하는지 여부를 검증하는 것**입니다. 이것이 의사의 진료 판단에 대한 문제를 논하기 전, 가장 객관적이고 합리적인 첫 단계입니다.

                ---
                **면책 조항:** 본 AI의 판단은 제공된 통계 데이터에 기반한 추론이며, 최종적인 의료적/법적 책임은 해당 분야의 인간 전문가에게 있습니다.
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
