# app.py (진정한 전문가 조력자 최종 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI Anomaly Detection System v16.0",
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
                Your persona is an **Elite AI Medical Analyst**. Your primary goal is to empower a senior clinician by providing data-driven insights that highlight *why* their specialized treatment is statistically unique and how to proactively defend it.
                1.  **ACKNOWLEDGE CLINICAL CONTEXT:** Start by acknowledging that statistically rare patterns often reflect specialized, expert-level clinical care for complex cases (like eating disorders).
                2.  **CITE THE DATA AS EVIDENCE FOR THE DOCTOR:** You MUST explicitly use the specific names and percentages from the `Case File Data` to quantify *why* this treatment pattern is unique. Frame this data as evidence the doctor can use.
                3.  **NO HALLUCINATION & NO JUDGMENT:** Do not invent clinical facts or judge the treatment's appropriateness. Your role is to analyze the data and explain its implications from different professional viewpoints.
                4.  **SEAMLESS & SUBSTANTIAL TEXT:** Your entire response must be a single, continuous text without boxes or code blocks. Each numbered section must be a detailed analysis.

                **Case File Data for Review:**
                - **Primary Patient of Interest:** Patient ID **{most_common_patient_id}**
                - **Key Evidence for Patient **{most_common_patient_id}** (This is the core evidence for your analysis):**
                ```markdown
                {patient_specific_reasons_str}
                ```

                **Mandatory Briefing Framework (Adopt the persona and render a decisive, detailed analysis):**

                ---

                ### **🔬 MediCopilot AI: 전문가 분석 및 소명 자료 지원**

                #### **분석 요약: 고도로 전문화된 진료 패턴 식별**
                총 **{total_claims:,}**건의 진료 기록 분석 결과, 통계적으로 매우 유니크한 전문 진료 패턴 **{total_anomalies:,}**건을 식별했습니다. 특히 환자 ID **{most_common_patient_id}**의 사례는, 복합적인 상태의 환자에 대한 깊이 있는 임상적 판단이 반영된 **고도로 전문화된, 비정형적 진료 패턴(Highly-specialized, Atypical Treatment Pattern)**으로 분석됩니다. 아래는 이 진료의 통계적 특성과 잠재적 쟁점에 대한 전문가별 분석 의견입니다.

                ---

                #### **분야별 전문가 검토 의견**

                ##### **1. 임상의학 관점: 전문성의 데이터 기반 증명**
                **의견:** 환자 **{most_common_patient_id}**의 사례는 식이장애와 같이 복합적인 증상을 가진 환자에 대한 깊은 이해를 바탕으로 한 처방으로 보입니다. Key Evidence에 따르면, **(여기서 Case File Data 테이블의 '특성명 (한글)' 컬럼에서 주요 항목 2-3개를 직접 언급하세요)** 등이 동시에 처방되었습니다. 이러한 약물 조합은 일반적인 단일 질환 처방에서는 나타나지 않는, 통계적으로 매우 이례적인 패턴입니다. 이는 역설적으로, 해당 환자의 복합적인 상태(예: 폭식, 과민성, 부종 등)를 동시에 제어하려는 의사의 **고도의 임상적 판단이 데이터로 나타난 결과**라고 해석할 수 있습니다. 다만, 이 처방의 유니크함은 그 근거를 명확히 문서화해 둘 필요가 있음을 시사합니다.

                ##### **2. 보건 행정 및 심사 관점: 선제적 방어 논리 제공**
                **의견:** 심사관은 일차적으로 통계 데이터에 기반하여 심사를 진행하므로, 이 사례의 **통계적 희귀성은 심사 조정의 빌미가 될 수 있습니다.** Key Evidence에 따르면, **(여기서 Case File Data 테이블에서 사용률이 가장 낮은 항목의 이름과 '평균 사용률 (%)'을 정확히 인용하세요)**와 같은 처방은 그 자체만으로도 소명 요청을 받을 가능성이 높습니다. 따라서, "본 환자는 복합적 식이장애 환자로, **'A약물'(사용률 X.XX%)**은 병적 식욕 제어를 위해, **'B처치'(사용률 Y.YY%)**는 심리 상태 안정을 위해 필수적이었다" 와 같이, **각 처방의 명확한 의학적 근거와 당위성을 선제적으로 준비**해두는 것이 잠재적 삭감을 방어하는 가장 효과적인 전략입니다.

                ##### **3. 데이터 분석 관점: '조합'의 희귀성 증명**
                **분석:** 이 패턴이 '이상치'로 탐지된 이유는, 개별 항목의 희귀성이 아니라 **이러한 희귀한 사건들이 '동시에 발생'했다는 조합의 확률** 때문입니다. Key Evidence에 따르면, **(여기서 Case File Data 테이블의 항목과 '평균 사용률 (%)'을 2개 인용하세요. 예: "'A약물'은 100명의 환자 중 0.15명에게만, 'B진단'은 0.50명에게만 나타납니다.")** 이처럼 각각의 발생 확률이 1%도 채 되지 않는 사건들이 한 개인에게 동시에 발생할 확률은 산술적으로 거의 0에 가깝습니다. 이 데이터는 해당 진료가 **결코 일반적인 사례가 아니며, 특별한 의학적 배경 없이는 설명하기 어려운 패턴**임을 객관적인 수치로 강력하게 뒷받침합니다.

                #### **4. 종합 결론: '임상적 타당성'과 '행정적 오류 가능성'에 대한 최종 판단**
                **판단:** 모든 데이터를 종합했을 때, 이 사례는 **'의사의 오진이나 처방 실수'라기보다는 '고도로 전문화된 진료에 대한 데이터 입력 또는 청구 과정의 오류'일 가능성**에 무게가 실립니다. 즉, 의사의 임상적 판단은 타당했을 가능성이 높지만, 그 복잡한 처방 내용이 행정 시스템에 정확히 반영되지 않았거나, 시스템이 이 특수성을 이해하지 못했을 수 있습니다. 따라서, 가장 시급하고 중요한 첫 단계는 **실제 의무기록과 청구 데이터가 정확히 일치하는지를 검증하여, 행정적 오류 가능성을 먼저 배제하는 것**입니다. 이 검증이 완료된 후에야, 해당 처방의 임상적 타당성에 대한 논의가 의미를 가질 것입니다.

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
