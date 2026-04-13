import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 頁面 ---
st.set_page_config(page_title="CKD 智能分析系統", layout="wide")

# --- CSS（手機優化🔥）---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f5f7fb; }

/* 卡片 */
.card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}

/* 標題 */
.card-title {
    font-size: 18px;
    font-weight: 600;
    color: #1f4e79;
}

/* KPI */
.kpi {
    font-size: 26px;
    font-weight: bold;
    color: #0a3d62;
}

/* 手機優化 */
@media (max-width: 768px) {
    .card { padding: 12px; }
    .kpi { font-size: 22px; }
}
</style>
""", unsafe_allow_html=True)

# --- 資料 ---
@st.cache_data
def load_data():
    return pd.read_csv("./data/CKD_cleaned.csv")

df = load_data()

# --- 模型 ---
model = None
scaler = None

if 'classification' in df.columns:

    model_df = df.copy()

    for col in ['age','bmi','serum_creatinine']:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    model_df = model_df.dropna()
    model_df['target'] = model_df['classification'].apply(lambda x: 1 if x=='ckd' else 0)

    X = model_df[['age','bmi','serum_creatinine']]
    y = model_df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

# --- Sidebar ---
st.sidebar.title("📱 CKD 系統")

page = st.sidebar.radio("導覽", ["Dashboard", "資料展示"])

# =========================
# 📊 Dashboard（手機優化）
# =========================
if page == "Dashboard":

    st.title("🏥 CKD 智能分析平台")

    # Tabs（手機好用🔥）
    tab1, tab2 = st.tabs(["📊 分析", "🧠 預測"])

    # =====================
    # 📊 Tab1 分析
    # =====================
    with tab1:

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
            st.markdown(f"""
            <div class="card">
            <div class="card-title">平均年齡</div>
            <div class="kpi">{round(df.age.mean(),1)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 圖表（手機自動直排🔥）
        st.markdown("### 📈 資料分析")

        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.regplot(data=df, x='age', y='serum_creatinine', ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =====================
    # 🧠 Tab2 預測
    # =====================
    with tab2:

        st.subheader("🧪 輸入資料")

        age = st.slider("年齡", 0, 100, 50)
        cre = st.number_input("肌酸酐", 0.0, 15.0, 1.2)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
        htn = st.checkbox("高血壓")
        dm = st.checkbox("糖尿病")

        if st.button("🔍 評估風險"):

            score = 0
            if age > 60: score += 1
            if cre > 1.5: score += 2
            if bmi > 30: score += 1
            if htn: score += 1
            if dm: score += 1

            st.markdown("### 📊 評估結果")

            if score >= 4:
                st.error(f"高風險（{score}）")
            elif score >= 2:
                st.warning(f"中風險（{score}）")
            else:
                st.success(f"低風險（{score}）")

            if model is not None:
                prob = model.predict_proba(scaler.transform([[age,bmi,cre]]))[0][1]
                st.info(f"CKD 機率：{round(prob*100,1)}%")

# =========================
# 📁 資料展示（手機優化🔥）
# =========================
elif page == "資料展示":

    st.title("📁 資料與成果")

    st.subheader("📋 資料預覽")

# --- 建立 target（如果不存在）---
    display_df = df.copy()

    if 'target' not in display_df.columns:
        if 'classification' in display_df.columns:
            display_df['target'] = display_df['classification'].apply(
                lambda x: 1 if str(x).lower() == 'ckd' else 0
            )
        else:
            display_df['target'] = None  # 避免報錯

    # --- 欄位順序 ---
    selected_cols = [
        'age',
        'bmi',
        'systolic_bp',
        'diastolic_bp',
        'serum_creatinine',
        'diabetes',
        'hypertension',
        'target'
    ]

    # --- 過濾存在欄位（避免 crash）---
    available_cols = [col for col in selected_cols if col in display_df.columns]

    # --- 顯示 ---
    st.dataframe(
        display_df[available_cols].head(20),
        use_container_width=True
    )

    st.divider()

    st.subheader("🖼️ 分析成果")

    image_paths = [
        "./screenshot/001.png",
        "./screenshot/002.png",
        "./screenshot/003.png",
        "./screenshot/004.png",
        "./screenshot/005.png",
        "./screenshot/006.png"
    ]

    titles = [
        "圖1：系統總覽",
        "圖2：資料分佈",
        "圖3：關聯分析",
        "圖4：模型預測",
        "圖5：特徵分析",
        "圖6：結論"
    ]

    # ⭐ 手機優化排列（關鍵🔥）
    for i in range(len(image_paths)):

        st.markdown(f"### {titles[i]}")

        try:
            st.image(image_paths[i], use_container_width=True)
        except:
            st.warning("圖片不存在")

        st.divider()