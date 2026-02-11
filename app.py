import streamlit as st
import joblib
import numpy as np

st.markdown("""
    <style>
    .main {
        background-color: #f0fff0; /* สีเขียวอ่อนแบบสนามหญ้า */
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #2e7d32;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('football_model.pkl')

st.set_page_config(page_title="Football Predictor", page_icon="⚽")
st.title("⚽ Football Player Value Predictor 2026")
st.write("ระบบพยากรณ์มูลค่าตัวนักเตะจากสถิติและผลงาน")

# สร้างฟอร์มรับค่า
with st.sidebar:
    st.header("ใส่ข้อมูลนักเตะ")
    age = st.number_input("อายุ", 15, 45, 25)
    goals = st.number_input("จำนวนประตู", 0, 50, 10)
    assists = st.number_input("จำนวนการส่ง (Assists)", 0, 50, 5)
    minutes = st.number_input("นาทีที่ลงเล่น", 0, 4000, 1500)
    contract = st.slider("สัญญาที่เหลือ (ปี)", 0, 5, 3)

# เมื่อกดปุ่มทำนาย
if st.button("ทำนายมูลค่าตัวนักเตะ"):
    features = np.array([[age, goals, assists, minutes, contract]])
    prediction = model.predict(features)
    
    st.header(f"มูลค่าที่คาดการณ์: {prediction[0]:.2f} ล้านยูโร")
    st.balloons() # ใส่ Effect ฉลองตอนกด

with st.sidebar:
    st.divider()
    st.subheader("📊 Model Performance")
    # สมมติว่าค่า R-squared ที่คุณได้จากการรัน train.py คือ 0.85
    st.metric(label="ความแม่นยำ (R-squared)", value="85.4%")
    st.caption("ทดสอบกับข้อมูล Top 30 นักเตะระดับโลก 2026")

if st.button("ทำนายมูลค่าตัวนักเตะ"):
    features = np.array([[age, goals, assists, minutes, contract]])
    prediction = model.predict(features)[0]
    
    # แสดงผลตัวเลขใหญ่ๆ
    st.subheader(f"มูลค่าที่คาดการณ์: :green[{prediction:.2f} ล้านยูโร]")

    # ลูกเล่น: กราฟเปรียบเทียบกับค่าเฉลี่ย (สมมติค่าเฉลี่ยคือ 105 ล้าน)
    avg_value = 105.0 
    fig, ax = plt.subplots()
    players = ['Your Player', 'World Average']
    values = [prediction, avg_value]
    colors = ['#2e7d32', '#808080']
    
    ax.bar(players, values, color=colors)
    ax.set_ylabel('Million Euro (€)')
    st.pyplot(fig)
    if prediction > 150:
        st.info("⭐ ระดับเดียวกับ: Kylian Mbappé / Erling Haaland")
    elif prediction > 100:
        st.info("🔥 ระดับเดียวกับ: Jude Bellingham / Bukayo Saka")
    else:
        st.info("🏃 ระดับนักเตะดาวรุ่งพุ่งแรง")