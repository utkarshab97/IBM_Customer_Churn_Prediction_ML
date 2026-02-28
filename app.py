import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
plt.rc('font', family='DejaVu Sans')
import io

# 1. 모델 및 자산 로드
@st.cache_resource # 모델 로딩 속도 최적화
def load_assets():
    model = joblib.load('models/churn_model_full.pkl')
    explainer = joblib.load('models/model_explainer.pkl')
    features = joblib.load('models/features_full.pkl')
    return model, explainer, features

model, explainer, features = load_assets()

def generate_explanation_plot(customer_df):
    """SHAP Waterfall Plot을 생성하고 웹에 표시할 수 있도록 변환합니다."""
    shap_values = explainer(customer_df)
    
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=5, show=False)
    
    # 이미지를 바이트로 변환하여 저장
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    return buf

# 2. 비즈니스 로직 함수
def get_business_advice(top_reasons):
    advice = []
    if 'Contract' in top_reasons:
        advice.append("**약정 전환 제안:** 월별 계약 고객입니다. 1년 약정 시 할인을 제안하여 이탈 장벽을 구축하세요.")
    if 'price_fatigue_index' in top_reasons:
        advice.append("**가격 피로도 관리:** 가입 기간 대비 요금 부담이 높습니다. '장기 고객 요금 동결' 프로그램을 제안하세요.")
    if 'OnlineSecurity' in top_reasons:
        advice.append("**서비스 번들링:** 보안 서비스가 없습니다. 3개월 무료 체험을 통해 서비스 의존도를 높이세요.")
    if 'MonthlyCharges' in top_reasons:
        advice.append("**요금제 최적화:** 요금 부담이 매우 큽니다. 경쟁사 이탈 전 선제적인 다운셀링을 고려하세요.")
    if not advice:
        advice.append("**일반 케어:** 정기적인 만족도 조사 및 웰컴 콜을 통해 관계를 유지하세요.")
    return advice

# 3. 사이드바 입력 폼 (Sidebar Input)
st.sidebar.header("고객 데이터 입력")

def user_input_features():
    # 1. 기본 인적 사항 및 서비스 이용 현황
    gender = st.sidebar.selectbox("성별 (gender)", ("Male", "Female"))
    senior = st.sidebar.selectbox("고령자 여부 (SeniorCitizen)", (0, 1))
    partner = st.sidebar.checkbox("파트너 여부 (Partner)")
    dependents = st.sidebar.checkbox("부양가족 여부 (Dependents)")
    tenure = st.sidebar.slider("가입 기간 (tenure)", 1, 72, 12)
    
    # 2. 계약 및 요금 정보
    contract = st.sidebar.selectbox("계약 형태 (Contract)", ("Month-to-month", "One year", "Two year"))
    paperless = st.sidebar.checkbox("전자 청구서 사용 (PaperlessBilling)")
    payment = st.sidebar.selectbox("결제 수단 (PaymentMethod)", 
                                  ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.sidebar.number_input("월 요금 (MonthlyCharges)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("총 요금 (TotalCharges)", 18.0, 9000.0, monthly_charges * tenure)

    # 3. 서비스 세부 항목
    phone = st.sidebar.checkbox("전화 서비스 (PhoneService)")
    multiple = st.sidebar.selectbox("다중 회선 (MultipleLines)", ("No", "Yes", "No phone service"))
    internet = st.sidebar.selectbox("인터넷 서비스 (InternetService)", ("DSL", "Fiber optic", "No"))
    security = st.sidebar.selectbox("온라인 보안 (OnlineSecurity)", ("No", "Yes", "No internet service"))
    backup = st.sidebar.selectbox("온라인 백업 (OnlineBackup)", ("No", "Yes", "No internet service"))
    protection = st.sidebar.selectbox("기기 보호 (DeviceProtection)", ("No", "Yes", "No internet service"))
    support = st.sidebar.selectbox("기술 지원 (TechSupport)", ("No", "Yes", "No internet service"))
    streaming_tv = st.sidebar.selectbox("스트리밍 TV (StreamingTV)", ("No", "Yes", "No internet service"))
    streaming_movies = st.sidebar.selectbox("스트리밍 영화 (StreamingMovies)", ("No", "Yes", "No internet service"))

    # 4. 커스텀 피처 계산 (Pipeline의 SQL 로직 재현)
    # Bundle Density 계산
    services = [phone, multiple == 'Yes', internet != 'No', security == 'Yes', 
                backup == 'Yes', protection == 'Yes', support == 'Yes', 
                streaming_tv == 'Yes', streaming_movies == 'Yes']
    bundle_density = sum(services)
    # Payment Friction 계산
    payment_friction = 1 if payment == 'Electronic check' else 0
    # Contract Leverage 계산
    if contract == 'Month-to-month' and monthly_charges > 70:
        leverage = 'High-Risk Flex'
    elif contract == 'Month-to-month':
        leverage = 'Low-Cost Flex'
    else:
        leverage = 'Contract Bound'
    # Price Fatigue 계산
    price_fatigue = monthly_charges / tenure if tenure > 0 else monthly_charges
    # Overpaying Flag 계산
    contract_thresholds = {
        "Month-to-month": 66.3,
        "One year": 65.0,
        "Two year": 60.8
    }
    overpaying_flag = 1 if monthly_charges > contract_thresholds.get(contract, 65.0) else 0
    # Unbalanced Bundle 계산
    essential_services = [security == 'Yes', backup == 'Yes', protection == 'Yes', 
                          support == 'Yes', streaming_tv == 'Yes', streaming_movies == 'Yes']
    essential_count = sum(essential_services)
    unbalanced_bundle = 1 if (monthly_charges > 70 and essential_count < 1) else 0

    # 5. 데이터프레임 생성 (features 리스트와 순서 일치)
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': int(partner), 'Dependents': int(dependents),
        'tenure': tenure, 'PhoneService': int(phone), 'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': int(paperless), 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'bundle_density': bundle_density, 'payment_friction_flag': payment_friction, 'overpaying_flag': overpaying_flag,
        'unbalanced_bundle': unbalanced_bundle, 'contract_leverage': leverage, 'price_fatigue_index': price_fatigue
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. 메인 화면 출력
st.title("Telco 고객 이탈 예측 및 분석 시스템")
st.write("사이드바에 고객 정보를 입력하면 이탈 위험도와 실시간 분석 결과를 제공합니다.")

if st.button('분석 실행'):
    # 데이터 타입 변환 (XGBoost 카테고리 대응)
    cat_cols = input_df.select_dtypes(include=['object']).columns
    input_df[cat_cols] = input_df[cat_cols].astype('category')
    
    # 예측
    try:
        # features에 정의된 순서대로 input_df의 컬럼을 재배열합니다.
        input_df = input_df[features]

        # 3. 예측 실행
        prob = model.predict_proba(input_df)[0][1]
        
        # 결과 요약
        col1, col2 = st.columns(2)
        with col1:
            st.metric("이탈 확률", f"{prob:.2%}")
        with col2:
            risk_status = "🔴 고위험" if prob > 0.7 else "🟡 주의" if prob > 0.4 else "🟢 안전"
            st.metric("위험 등급", risk_status)

        # SHAP Waterfall Plot
        st.subheader("주요 이탈 원인 분석 (Explainable AI)")
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=5, show=False)
        st.pyplot(fig)

        # 비즈니스 권고안
        st.subheader("💡 맞춤형 리텐션 전략")
        # SHAP 기여도가 높은 상위 3개 변수 추출
        feature_impacts = pd.Series(shap_values.values[0], index=features)
        top_reasons = feature_impacts.sort_values(ascending=False).head(3).index.tolist()

        for advice in get_business_advice(top_reasons):
            st.info(advice)
    except KeyError as e:
        st.error(f"피처 불일치 에러: 모델에 필요한 {e} 컬럼이 입력 데이터에 없습니다.")
        st.write("모델이 요구하는 컬럼 순서:", features)