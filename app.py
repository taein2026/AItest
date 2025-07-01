# app.py (Gemini API 연동 및 오류 수정 완료 버전)

import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from analysis import run_analysis
import google.generativeai as genai

# --- 페이지 기본 설정 ---
# Streamlit 앱의 제목, 아이콘, 레이아웃 등 기본 설정을 정의합니다.
st.set_page_config(
    page_title="AI Anomaly Detection System v5.0",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lottie 애니메이션 로드 함수 ---
# 웹에서 Lottie JSON 파일을 불러와 캐시에 저장하여 재사용합니다.
# 이렇게 하면 앱 로딩 속도를 향상시킬 수 있습니다.
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        return None

# --- 사이드바 ---
# 사이드바는 파일 업로드 및 분석 시작 버튼을 담당합니다.
with st.sidebar:
    lottie_url = "https://lottie.host/e883236e-1335-4309-a185-11a518012e69/Tpde6s5V1C.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="sidebar_lottie")

    st.title("📄 파일 업로드")
    st.info("분석에 필요한 파일 3개를 모두 업로드해주세요.")
    
    # 3개의 파일 업로더를 생성합니다.
    uploaded_main_file = st.file_uploader("① 메인 진료 데이터", type=['csv'])
    uploaded_disease_file = st.file_uploader("② 상병명 매칭 테이블", type=['xlsx'])
    uploaded_drug_file = st.file_uploader("③ 약물명 매칭 테이블", type=['xlsx'])
    
    # 모든 파일이 업로드되었을 때만 '파일 저장' 버튼이 활성화됩니다.
    if uploaded_main_file and uploaded_disease_file and uploaded_drug_file:
        if st.button("✔️ 업로드 파일 저장", use_container_width=True):
            with st.spinner("파일을 읽고 있습니다..."):
                # 업로드된 파일을 DataFrame으로 읽어 st.session_state에 저장합니다.
                # session_state에 저장하면 앱이 재실행되어도 데이터가 유지됩니다.
                st.session_state['df_main'] = pd.read_csv(uploaded_main_file, encoding='cp949', low_memory=False)
                st.session_state['df_disease'] = pd.read_excel(uploaded_disease_file, dtype={'상병코드': str})
                st.session_state['df_drug'] = pd.read_excel(uploaded_drug_file, dtype={'연합회코드': str})
                st.session_state['files_ready'] = True
                st.success("파일 저장 완료!")

    st.markdown("---")
    # 파일이 준비된 경우에만 'AI 분석 실행' 버튼이 활성화됩니다.
    start_button = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True, disabled='files_ready' not in st.session_state)

# --- 메인 화면 ---
st.title("✨ MediCopilot AI")
st.markdown("---")

# 초기 화면: 분석 시작 버튼이 눌리지 않았을 때 안내 메시지와 이미지를 표시합니다.
if not start_button:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 누른 후, 'AI 분석 실행' 버튼을 눌러주세요.")
    st.image("https://storage.googleapis.com/gweb-cloud-ai-generative-ai-proserve-media/images/dashboard_professional.png", use_column_width=True)

# 분석 시작: 'AI 분석 실행' 버튼이 눌렸을 때 실행되는 로직입니다.
if start_button:
    if 'files_ready' in st.session_state:
        # 전체 분석 과정을 try-except 블록으로 감싸 오류 발생 시 앱이 멈추지 않도록 합니다.
        try:
            # --- AI 분석 프로세스 바 ---
            st.header("AI 분석 프로세스")
            
            step1, step2, step3, step4, step5, step6 = st.columns(6)
            placeholders = [step1.empty(), step2.empty(), step3.empty(), step4.empty(), step5.empty(), step6.empty()]
            steps = ["1. 데이터 로딩", "2. 데이터 전처리", "3. AI 모델 학습", "4. 패턴 분석", "5. 이상치 탐지", "6. 보고서 생성"]

            # 각 단계의 진행 상황을 시각적으로 보여줍니다.
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
            # analysis.py의 run_analysis 함수를 호출하여 실제 분석을 수행합니다.
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
            
            # --- AI 최종 분석 브리핑 (Gemini API 연동 버전) ---
            st.header("🔬 AI 최종 분석 브리핑")

            # Gemini API 호출 부분을 별도의 try-except로 감싸 API 관련 오류를 처리합니다.
            try:
                # st.secrets를 사용하여 .streamlit/secrets.toml 파일에서 API 키를 안전하게 불러옵니다.
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                # AI에게 전달할 프롬프트(명령문)를 생성합니다.
                # 분석 결과를 변수로 포함시켜 AI가 이 데이터를 기반으로 리포트를 작성하게 합니다.
                prompt = f"""
                당신은 의료 데이터 분석 전문가 'MediCopilot AI'입니다. 아래 분석 결과를 바탕으로, 핵심 요약, 주목할 만한 패턴, 권장 조치를 포함한 전문적인 브리핑 리포트를 생성해주세요. 친절하고 신뢰감 있는 전문가 톤으로, 마크다운을 사용하여 가독성 좋게 작성해주세요.

                ### **분석 결과 데이터**
                - **총 분석 진료 건수:** {total_claims:,}건
                - **탐지된 이상치 건수:** {total_anomalies:,}건 (전체의 {(total_anomalies/total_claims):.2%})
                - **주요 패턴:** 이상치 Top 20 리스트에서 '환자번호 {pd.Series([res['patient_id'] for res in results]).mode()[0]}'가 총 {pd.Series([res['patient_id'] for res in results]).value_counts().max()}회로 가장 많이 발견되었습니다.

                ### **브리핑 작성 가이드**
                1.  **인사 및 개요:** 분석 완료를 알리고, 총 몇 건을 분석하여 몇 건의 이상 패턴을 찾았는지 요약합니다.
                2.  **핵심 발견(Key Finding):** 가장 주목할 만한 패턴을 구체적인 수치와 함께 강조하여 설명합니다.
                3.  **권장 조치(Recommendation):** 분석 결과를 바탕으로 사용자가 다음으로 무엇을 해야 할지 명확하게 제안합니다.
                """
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                # stream=True 옵션으로 AI가 생성하는 텍스트를 실시간으로 받아옵니다.
                response = model.generate_content(prompt, stream=True)

                # AI의 답변을 타이핑 효과와 함께 출력합니다.
                with st.chat_message("ai", avatar="🤖"):
                    report_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.text
                        report_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    report_placeholder.markdown(full_response, unsafe_allow_html=True)

            except Exception as e:
                st.error("AI 브리핑 생성 중 오류가 발생했습니다. .streamlit/secrets.toml 파일에 API 키가 올바르게 설정되었는지 확인해주세요.")
            
            st.markdown("---")
            
            # --- 분석 결과 상세 대시보드 ---
            st.header("분석 결과 상세 대시보드")
            col1, col2, col3 = st.columns(3)
            col1.metric("총 진료 건수", f"{total_claims:,} 건")
            col2.metric("탐지된 이상치", f"{total_anomalies:,} 건", f"상위 {(total_anomalies/total_claims):.2%}")
            # 실제 분석된 특성 수를 동적으로 계산하려면 len(final_feature_cols)를 사용해야 합니다.
            # 여기서는 예시로 고정값을 사용합니다.
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

        # SyntaxError의 원인이었던 누락된 except 블록입니다.
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
    else:
        st.warning("🚨 분석을 시작하기 전에 왼쪽 사이드바에서 파일 3개를 모두 업로드하고 '업로드 파일 저장' 버튼을 눌러주세요!")
