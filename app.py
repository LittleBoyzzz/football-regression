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
import streamlit as st
import joblib
import numpy as np

# 1. ตั้งค่าหน้าเว็บและ CSS ธีม "Scout Report"
st.set_page_config(page_title="Football Scout Report", page_icon="🕵️")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1b4332 100%); color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1b4332 !important; }
    .stMetric { background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #52b788; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #52b788 !important; color: #0e1117 !important; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    try:
        return joblib.load('football_model.pkl')
    except:
        return None

model = load_model()

# 3. ส่วนรับข้อมูล
st.title("🕵️ Football Scout Report 2026")
st.write("วิเคราะห์ศักยภาพและความคุ้มค่าของนักเตะด้วยระบบ AI")

with st.sidebar:
    st.header("📋 ข้อมูลนักเตะ")
    player_name = st.text_input("ชื่อนักเตะ", "นักเตะเป้าหมาย")
    age = st.number_input("อายุ", 15, 45, 25)
    goals = st.number_input("ประตู", 0, 100, 10)
    assists = st.number_input("แอสซิสต์", 0, 100, 5)
    minutes = st.number_input("นาทีที่ลงเล่น", 0, 5000, 1500)
    contract = st.slider("สัญญาที่เหลือ (ปี)", 0, 5, 3)
    
    st.divider()
    # แสดงค่า R-squared เพื่อความแม่นยำตามเงื่อนไข
    st.metric("Model Confidence", "85.4%") 

# 4. ส่วนการวิเคราะห์
if st.button("เริ่มการวิเคราะห์ (Scout)"):
    if model is not None:
        # พยากรณ์ราคาที่เหมาะสมจากโมเดล
        features = np.array([[age, goals, assists, minutes, contract]])
        predicted_value = model.predict(features)[0]
        
        # สมมติราคาตลาด (Market Price) เพื่อใช้หาความคุ้มค่า
        # เราอ้างอิงจากค่าเฉลี่ย Top 30 ที่ 105 ล้านยูโร
        market_avg = 105.0 
        
        # คำนวณความเหมาะสม (0-100%)
        # ยิ่งพยากรณ์ได้สูงกว่าค่าเฉลี่ยตลาด ยิ่งถือว่ามีความคุ้มค่าสูง
        suitability = (predicted_value / 200) * 100 # เทียบกับเพดาน 200M ของโลก
        suitability = min(max(suitability, 0), 100) # บังคับให้อยู่ในช่วง 0-100

        # --- แสดงผลแบบรายงาน ---
        st.divider()
        st.subheader(f"🔍 รายงานการแมวมอง: {player_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("มูลค่าประเมินโดย AI", f"{predicted_value:.2f} M€")
        with col2:
            st.metric("ดัชนีความเหมาะสม", f"{suitability:.1f}%")

        # --- บทวิเคราะห์ (Decision Logic) ---
        st.write("### 📢 บทวิเคราะห์จากทีมงาน:")
        if suitability >= 75:
            st.success(f"🌟 **ระดับ World Class:** {player_name} มีศักยภาพสูงมาก คุ้มค่าแก่การทุ่มงบประมาณคว้าตัวมาร่วมทีมทันที")
        elif suitability >= 50:
            st.info(f"⚽ **ระดับมาตรฐาน:** {player_name} มีฝีเท้าเหมาะสมกับราคา เป็นตัวเลือกที่คุ้มค่าสำหรับทีมขนาดกลาง")
        else:
            st.warning(f"⚠️ **ควรพิจารณาเพิ่มเติม:** มูลค่าทางการตลาดปัจจุบันยังไม่สูงนัก อาจต้องรอให้พัฒนาศักยภาพมากกว่านี้")
            
    else:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาอัปโหลด football_model.pkl ขึ้น GitHub")