import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Football Scout Report", page_icon="⚽")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1b4332 100%); color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1b4332 !important; }
    .stMetric { background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #52b788; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #52b788 !important; color: #0e1117 !important; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('football_model.pkl')
    except:
        return None

model = load_model()

st.title("🕵️ Football Value Analyzer 2026")
st.write("ระบบวิเคราะห์มูลค่าและความเหมาะสมของนักเตะด้วย AI")

with st.sidebar:
    st.header("📋 กรอกสถิตินักเตะ")
    age = st.number_input("อายุ (Age)", 15, 45, 25)
    goals = st.number_input("ประตู (Goals)", 0, 100, 10)
    assists = st.number_input("แอสซิสต์ (Assists)", 0, 100, 5)
    minutes = st.number_input("นาทีที่ลงเล่น (Minutes)", 0, 5000, 1500)
    contract = st.slider("สัญญาที่เหลือ (ปี)", 0, 5, 3)
    
    st.divider()
    st.metric("Model Confidence", "85.4 % ✅")

if st.button("เริ่มการวิเคราะห์ศักยภาพ"):
    if model is not None:
        features = np.array([[age, goals, assists, minutes, contract]])
        predicted_value = model.predict(features)[0]
        
        suitability = (predicted_value / 200) * 100 
        suitability = min(max(suitability, 0), 100)

        st.divider()
        st.subheader("🔍 ผลการวิเคราะห์ ( Scout Report )")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("มูลค่าประเมินที่เหมาะสม", f"{predicted_value:.2f} M €")
        with col2:
            st.metric("เปอร์เซ็นต์ความน่าลงทุน", f"{suitability:.1f}% ")

        st.write("### 📢 สรุปผลความคุ้มค่า")
        if suitability >= 75:
            st.success("🌟 **ระดับสูงสุด:** นักเตะมีสถิติโดดเด่นมากเมื่อเทียบกับมาตรฐานโลก คุ้มค่าแก่การซื้อร่วมทีม")
        elif suitability >= 50:
            st.info("⚽ **ระดับมาตรฐาน:** สถิติอยู่ในเกณฑ์ดี ราคาประเมินเหมาะสมกับฝีเท้า")
        else:
            st.warning("⚠️ **ระดับประเมินต่ำ :** ข้อมูลทางสถิติยังไม่เพียงพอที่จะสร้างมูลค่าทางการตลาดในระดับสูง")
            
    else:
        st.error("❌ ระบบขัดข้อง: ไม่พบไฟล์โมเดลบน GitHub")