# app.py (신뢰도 및 가독성 개선 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v6.0",
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
            
            # (진행 바 코드는 이전과 동일)
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
            
            # (진행 바 마무리)
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
                # 통계 수치를 AI에게 전달하기 전, 파이썬에서 퍼센트(%) 형식으로 미리 변환합니다.
                most_common_patient_id = pd.Series([res['patient_id'] for res in results]).mode()[0]
                patient_specific_reasons_str = "상세 정보 없음"
                for res in results:
                    if res['patient_id'] == most_common_patient_id:
                        reasons_df = res['reasons'].copy()
                        # '평균 사용률 (%)' 컬럼을 새로 만들어 보기 좋게 포맷팅합니다.
                        reasons_df['평균 사용률 (%)'] = (reasons_df['평균 사용률'] * 100).map('{:.2f}%'.format)
                        # AI에게 전달할 컬럼만 선택하여 Markdown 문자열로 변환합니다.
                        patient_specific_reasons_str = reasons_df[['특성명 (한글)', '평균 사용률 (%)']].to_markdown(index=False)
                        break

                # --- 2. AI 길들이기 (Advanced Prompt Engineering) ---
                prompt = f"""
                **CRITICAL INSTRUCTION: Your analysis MUST be based *strictly* on the `Input Data` provided. Do not use external medical knowledge to make definitive statements about drug interactions or clinical appropriateness. Instead, point out the statistical rarity and recommend that a human expert verify the clinical details. Your primary function is to interpret the data, not to act as a clinician.**

                **Your Role & Goal:**
                You are 'MediCopilot AI', a Multi-Disciplinary Medical AI Reviewer. Your role is to synthesize the provided data into a structured, objective report for a hospital's internal review committee.

                **Input Data:**
                - **Total Claims Analyzed:** {total_claims:,}
                - **Anomalous Patterns Detected:** {total_anomalies:,}
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**.
                - **Data for Patient **{most_common_patient_id}** (The '평균 사용률 (%)' column shows how often this item appears across all data):**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework:**
                Generate a briefing in Korean. You MUST follow this structure precisely.

                ---

                ### 🔬 MediCopilot AI 다학제 통합 분석 보고서

                #### **1. 분석 개요 (Executive Summary)**
                * `Input Data`에 명시된 총 진료 건수, 이상 패턴 식별 건수, 그리고 주요 관심 환자를 요약합니다.

                #### **2. 심층 분석: 주요 관심 환자 (**{most_common_patient_id}**)**
                * 이 환자가 통계적으로 왜 분석의 핵심 대상이 되었는지 `Input Data`에 근거하여 설명합니다.

                #### **3. 다각적 전문가 의견 (Multi-Faceted Expert Analysis)**
                * **3.1. 임상 및 규제 관점 (Clinical & Regulatory Perspective):**
                    * **(데이터 기반 서술)** '상세 이상 패턴 보고서'에 따르면, 이 환자는 사용률이 매우 낮은 항목들의 조합으로 치료받았습니다.
                    * **(전문가에게 질문 제기)** 이처럼 통계적으로 희귀한 조합의 의학적 타당성과 표준 진료 가이드라인 부합 여부는 반드시 임상 전문가의 검토가 필요합니다. 이 처방이 건강보험 심사 시 어떤 쟁점을 유발할 수 있는지 검토가 요구됩니다.

                * **3.2. 데이터 과학자 관점 (Data Science Perspective):**
                    * **(명확한 통계 제시)** 이 패턴이 통계적 '이상치'로 탐지된 이유는 `Input Data`의 '평균 사용률 (%)' 수치가 명확히 보여줍니다.
                    * `Input Data`의 '평균 사용률 (%)'을 직접 인용하여, 해당 조합이 전체 데이터에서 얼마나 드물게 발생하는지 설명하세요. (예: "'A 약물'은 전체 진료에서 단 **0.15%**만 사용된 희귀 처방입니다.")

                #### **4. 근본 원인에 대한 데이터 기반 추론 (Data-Driven Root Cause Hypothesis)**
                * **(메타인지적 접근)** 제공된 데이터만으로 근본 원인을 확정할 수는 없지만, 다음과 같은 가설을 세우고 검토를 제안할 수 있습니다.
                    * **가설 A (의료적 특이성):** 환자의 상태가 매우 특수하여 이례적인 처방이 필요했을 가능성. 이는 `Input Data`만으로는 검증이 불가능하며, 상세한 의무기록 확인이 필요합니다.
                    * **가설 B (행정적 오류):** 청구 코드 입력 실수 등의 오류 가능성. 데이터의 통계적 희귀성이 매우 높을 경우, 이 가설의 검토 우선순위가 높아질 수 있습니다.

                #### **5. 최종 권고 및 제언 (Final Recommendations & Limitations)**
                * 1. **(의무기록 대조)** 환자 **{most_common_patient_id}**의 상세 의무기록을 `Input Data`와 대조하여 사실관계를 확인해야 합니다.
                * 2. **(전문가 검토 의뢰)** 통계적으로 이례적인 이 처방 조합의 의학적 적정성에 대한 임상 전문가의 공식적인 검토를 요청해야 합니다.
                * 3. **(데이터 입력 프로세스 점검)** 행정적 오류 가능성을 배제하기 위해, 해당 진료 건의 데이터 입력 과정을 점검할 것을 권장합니다.
                * **(명확한 한계 고지)** **본 AI 보고서는 통계적 이상 패턴을 식별하는 보조 도구이며, 의료 행위의 적정성을 최종 판단하지 않습니다. 모든 결론은 반드시 해당 분야의 인간 전문가에 의해 검토되고 확증되어야 합니다.**

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
            
            # --- 대시보드 및 상세 분석 테이블 (이전과 동일) ---
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
                    expander_title = f"**Rank {res['rank']}** | 환자번호: `{res['patient_id']}` | 진료일: `{res['date']}`"
                    with st.expander(expander_title):
                        st.write("▶ **이 진료가 이상치로 판단된 핵심 이유 (가장 희귀한 조합 Top 5):**")
                        st.dataframe(res['reasons'], use_container_width=True) 

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 눌러주세요!")
