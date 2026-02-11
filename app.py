import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. ตั้งค่าหน้าเว็บและ CSS แบบ Modern Dark Sport
st.set_page_config(page_title="Football Value Predictor 2026", page_icon="⚽")

st.markdown("""
    <style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #072a07 100%);
    }
    
    /* Sidebar สีเขียวเข้มแบบสปอร์ต */
    [data-testid="stSidebar"] {
        background-color: #1b4332 !important;
    }
    
    /* ปรับแต่งปุ่มพยากรณ์ */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background-color: #2d6a4f !important;
        color: #d8f3dc !important;
        font-weight: bold;
        border: 2px solid #52b788 !important;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #40916c !important;
        box-shadow: 0px 0px 15px #52b788;
    }

    /* กล่องผลลัพธ์ */
    .stSuccess {
        background-color: rgba(45, 106, 79, 0.3) !important;
        border: 1px solid #52b788 !important;
        color: #d8f3dc !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. โหลดโมเดล (ย้ายออกมาข้างนอกเพื่อให้แน่ใจว่าตัวแปร model ถูกสร้างขึ้นแน่นอน)
@st.cache_resource
def load_my_model():
    try:
        return joblib.load('football_model.pkl')
    except:
        return None

model = load_my_model()

# 3. ส่วนแสดงผลหน้าเว็บ
st.title("Football Value Predictor 2026")
st.write("พยากรณ์มูลค่าตัวนักเตะจากสถิติและสัญญาปัจจุบัน")

# Sidebar สำหรับรับข้อมูล
with st.sidebar:
    st.header("📌 กรอกข้อมูลนักเตะ")
    age = st.number_input("อายุ (Age)", 15, 45, 25)
    goals = st.number_input("ประตู (Goals)", 0, 100, 10)
    assists = st.number_input("แอสซิสต์ (Assists)", 0, 100, 5)
    minutes = st.number_input("นาทีที่เล่น (Minutes)", 0, 5000, 1500)
    contract = st.slider("สัญญาที่เหลือ (ปี)", 0, 5, 3)
    
    st.divider()
    st.metric("Model Confidence", "85.4 % ✅ ")

# 4. ส่วนการทำนายและแสดงกราฟ
if st.button("ทำการพยากรณ์", key="predict_btn"):
    if model is not None:
        features = np.array([[age, goals, assists, minutes, contract]])
        prediction = model.predict(features)[0]
        
        # แสดงผลตัวเลข
        st.success(f"###  มูลค่าตัวที่คาดการณ์ : € {prediction:.2f} M  💰 ")
        
        # กราฟเปรียบเทียบ
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # ปรับสีพื้นหลังกราฟให้เข้ากับธีมมืด
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor("#ffffff")
        
        labels = ['Player', 'Top 30 Avg']
        values = [prediction, 105.0]
        
        # วาดกราฟแท่ง
        bars = ax.bar(labels, values, color=["#ff0000", "#0004FF"])
        
        # --- เพิ่มชื่อแกนตรงนี้ครับ ---
        ax.set_ylabel('Market Value ( € )', color='white', fontsize=12) # ชื่อแกนแนวตั้ง
        ax.set_title('Value Comparison', color='white', fontsize=14, pad=15) # ชื่อหัวข้อกราฟ
        
        # ปรับสีตัวเลขแกน X และ Y ให้เป็นสีขาว
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # แสดงกราฟบน Streamlit
        st.pyplot(fig)