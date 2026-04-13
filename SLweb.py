import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 避免 matplotlib 錯誤
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 頁面設定 ---
st.set_page_config(page_title="CKD 智能分析系統", layout="wide")

# --- 側邊導覽 ---
page = st.sidebar.radio("📂 導覽", ["📊 分析儀表板", "📁 資料與成果展示"])

# --- CSS ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f5f7fb; }

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.card-title {
    font-size: 18px;
    font-weight: 600;
    color: #1f4e79;
}

.kpi {
    font-size: 28px;
    font-weight: bold;
    color: #0a3d62;
}
</style>
""", unsafe_allow_html=True)

# --- 資料 ---
@st.cache_data
def load_data():
    return pd.read_csv("./data/CKD_cleaned.csv")

df = load_data()

# --- 建立模型 ---
model = None
scaler = None

if 'classification' in df.columns:

    model_df = df.copy()

    for col in ['age','bmi','serum_creatinine']:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    model_df = model_df.dropna()

    model_df['target'] = model_df['classification'].apply(
        lambda x: 1 if x == 'ckd' else 0
    )

    X = model_df[['age','bmi','serum_creatinine']]
    y = model_df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

# =========================
# 📊 頁面 1：Dashboard
# =========================
if page == "📊 分析儀表板":

    st.title("🏥 CKD 慢性腎臟病智能分析平台")
    st.caption("整合臨床數據、風險評估與機器學習預測")

    # --- Sidebar ---
    st.sidebar.header("🧪 病患輸入")

    age = st.sidebar.slider("年齡", 0, 100, 50)
    cre = st.sidebar.number_input("肌酸酐", 0.0, 15.0, 1.2)
    bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)
    htn = st.sidebar.checkbox("高血壓")
    dm = st.sidebar.checkbox("糖尿病")

    st.sidebar.divider()

    # --- 風險評估 ---
    st.sidebar.subheader("🧠 CKD 風險評估")

    if st.sidebar.button("🔍 評估CKD風險"):

        risk_score = 0
        if age > 60: risk_score += 1
        if cre > 1.5: risk_score += 2
        if bmi > 30: risk_score += 1
        if htn: risk_score += 1
        if dm: risk_score += 1

        if risk_score >= 4:
            st.sidebar.error(f"⚠️ 高風險（Score: {risk_score}）")
        elif risk_score >= 2:
            st.sidebar.warning(f"⚠️ 中度風險（Score: {risk_score}）")
        else:
            st.sidebar.success(f"✅ 低風險（Score: {risk_score}）")

        if model is not None:
            input_data = scaler.transform([[age, bmi, cre]])
            prob = model.predict_proba(input_data)[0][1]

            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🤖 模型預測")

            st.sidebar.info(f"CKD 機率：{round(prob*100,1)}%")

    # --- KPI ---
    st.subheader("📊 關鍵指標")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">樣本數</div>
            <div class="kpi">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_age = round(df['age'].mean(), 1)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">平均年齡</div>
            <div class="kpi">{avg_age} 歲</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- 圖表 ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="card-title">年齡分佈</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">Creatinine vs Age</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='age', y='serum_creatinine', ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Feature Importance ---
    if model is not None:

        coef = model.coef_[0]
        features = ['age','bmi','serum_creatinine']

        importance_df = pd.DataFrame({
            'feature': features,
            'importance': coef
        })

        st.markdown('<div class="card"><div class="card-title">影響因子分析</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots()
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Insight ---
    st.markdown("""
    <div class="card">
    <div class="card-title">📊 分析結論</div>

    - 肌酸酐為最重要風險因子  
    - 年齡與腎功能呈正相關  
    - BMI 對風險有次要影響  

    👉 建議優先監測腎功能與慢性病史
    </div>
    """, unsafe_allow_html=True)


# =========================
# 📁 頁面 2：資料 + 截圖
# =========================
elif page == "📁 資料與成果展示":

    st.title("📁 資料與分析成果展示")

    # --- 資料 ---
    st.subheader("📋 CKD 資料預覽")

    try:
        selected_cols = [
            col for col in ['age','bmi','systolic_bp','diastolic_bp','serum_creatinine','diabetes','hypertension','target']
            if col in df.columns
        ]

        st.dataframe(df[selected_cols].head(100), use_container_width=True)

    except:
        st.error("❌ 無法讀取資料")

    st.divider()

    # --- 圖片 ---
    st.subheader("🖼️ 分析成果截圖")

    image_paths = [
        "./screenshot/001.png",
        "./screenshot/002.png",
        "./screenshot/003.png",
        "./screenshot/004.png",
        "./screenshot/005.png",
        "./screenshot/006.png"
    ]

    titles = [
        "圖1：資料總覽",
        "圖2：資料分佈分析",
        "圖3：生理指標與CKD關聯",
        "圖4：高低風險族群分析",
        "圖5：年齡和CKD的趨勢",
        "圖6：BMI指標對CKD的影響"
    ]

for row in range(0, len(image_paths), 3):

    cols = st.columns(3)

    for col_idx in range(3):
        i = row + col_idx

        if i < len(image_paths):
            with cols[col_idx]:
                try:
                    st.markdown(f"**{titles[i]}**")
                    st.image(image_paths[i], use_container_width=True)
                except:
                    st.warning(f"⚠️ 找不到 {image_paths[i]}")