import streamlit as st
import joblib
import numpy as np

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