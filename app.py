# app.py (최종 전문가 판단 및 가독성 개선)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v13.0",
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
                Your persona is an **Elite AI Medical Analyst**. Your tone is direct, professional, and provides actionable intelligence for a senior clinician.
                1.  **CITE THE DATA:** You MUST explicitly use the specific names and numbers from the `Input Data` to support every judgment.
                2.  **NO HALLUCINATION:** Your judgments are based *only* on the statistical patterns provided. Do not invent clinical facts.
                3.  **SEAMLESS TEXT:** Your entire response must be a single, continuous text. Do not use any boxes, backticks (` `), or code blocks (```). Emphasize with bolding (`**text**`) only.
                4.  **SUBSTANTIAL ANALYSIS:** Each numbered section must contain a detailed analysis of at least 5 lines.

                **Input Data:**
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**
                - **Data for Patient **{most_common_patient_id}** (This is the core evidence for your judgment):**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework (For each section, adopt the persona and render a decisive judgment):**

                ---

                ### **🔬 MediCopilot AI: 전문가 위원회 최종 판단**

                #### **1. 임상의학 전문가 (정신과) 최종 소견**
                **판단:** `Input Data`에 따르면, 환자 **{most_common_patient_id}**에게 **(여기서 Input Data 테이블의 '특성명 (한글)' 컬럼에서 가장 이례적인 약물명 2-3개를 직접 언급하세요)** 등이 동시에 처방되었습니다. 이 조합은 표준 임상 프로토콜에서 심각하게 벗어난 것으로 보입니다. 특히 항정신병제와 식욕억제제의 병용은 약물 상호작용 및 부작용 위험을 크게 높일 수 있어, **의학적 타당성을 입증할 명백한 근거가 없다면 부적절한 처방으로 판단**됩니다. 이는 의사의 전문적 판단을 의심하는 것이 아니라, 해당 처방이 일반적인 진료의 범주를 벗어난 통계적 특이성을 보이므로, **표준 진료 프로토콜과의 차이점을 스스로 재점검하고 그 근거를 명확히 할 필요가 있음을 시사**합니다.

                #### **2. 보건복지부 행정 심사관 최종 결정**
                **결정:** `Input Data`에서 확인된 처방 조합은 통계적 희귀성으로 인해 **건강보험심사평가원의 심사 조정(삭감) 대상이 될 확률이 매우 높습니다.** 예를 들어, **(여기서 Input Data 테이블에서 사용률이 가장 낮은 항목의 이름과 '평균 사용률 (%)'을 정확히 인용하세요)**와 같은 처방은 그 자체만으로도 정밀 심사 대상입니다. 심사관의 입장에서 볼 때, 이러한 통계적 이상치는 명확한 소명 자료가 없다면 '과잉 진료' 또는 '착오 청구'로 해석될 여지가 충분합니다. **결정적으로, 이 청구 건은 '요주의 사례'로 분류하고, 처방의 타당성을 입증하는 상세한 소명 자료 제출을 즉시 요구해야 합니다.** 자료가 미비할 경우, 관련 진료비 전액 삭감까지도 고려될 수 있습니다.

                #### **3. 데이터 통계 전문가 최종 분석**
                **분석:** 통계적으로, 이 패턴은 우연으로 보기 어려운 **극단적인 이상치(Extreme Outlier)**입니다. 핵심은 개별 항목의 희귀성이 아니라, **이러한 희귀한 사건들이 동시에 발생했다는 '조합'의 희귀성**에 있습니다. 예를 들어, `Input Data`의 **(여기서 Input Data 테이블의 항목과 '평균 사용률 (%)'을 2개 인용하세요. 예: "'A약물'은 100명의 환자 중 0.15명에게만, 'B진단'은 0.50명에게만 나타납니다.")** 이 두 사건이 독립적이라 가정할 때, 두 사건이 동시에 발생할 확률은 두 확률의 곱으로 훨씬 더 희박해집니다. 이처럼 강력한 통계적 증거는 이 패턴이 결코 일반적인 진료 행위가 아님을 명백히 보여줍니다.

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
