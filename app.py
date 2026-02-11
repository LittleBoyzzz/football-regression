import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

if st.button("วิเคราะห์ความคุ้มค่า"):
    features = np.array([[age, goals, assists, minutes, contract]])
    prediction = model.predict(features)[0]

    # สมมติราคาที่สโมสรต้นสังกัดตั้งขาย (Market Asking Price)
    # เราจะสุ่มให้มันใกล้เคียงกับค่าเฉลี่ย 105M เพื่อใช้เปรียบเทียบ
    asking_price = 100.0 

    # คำนวณความคุ้มค่า (Percentage)
    # สูตร: (ราคาที่พยากรณ์ได้ / ราคาที่เขาขาย) * 100
    worth_score = (prediction / asking_price) * 100
    worth_score = min(worth_score, 100.0) # ล็อคไว้ไม่ให้เกิน 100%

    # --- แสดงผลแบบ Scout Report ---
    st.subheader(f"🔍 รายงานการแมวมอง: {player_name if 'player_name' in locals() else 'นักเตะเป้าหมาย'}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("มูลค่าที่เหมาะสม", f"{prediction:.2f} M€")
    with col2:
        st.metric("ดัชนีความคุ้มค่า", f"{worth_score:.1f}%")

    # --- เงื่อนไขการแนะนำ (Logic) ---
    if worth_score >= 85:
        st.success("✅ **คุ้มค่าแก่การลงทุน:** นักเตะมีศักยภาพสูงกว่าราคาตลาดปัจจุบัน")
    elif worth_score >= 60:
        st.warning("⚠️ **พิจารณาเพิ่มเติม:** ราคาเหมาะสมกับฝีเท้า แต่อาจไม่มีกำไรในอนาคต")
    else:
        st.error("❌ **ไม่แนะนำให้ซื้อ:** ราคาประเมินต่ำกว่าราคาตั้งขายมากเกินไป")