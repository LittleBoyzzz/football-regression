import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. ตั้งค่าหน้าเว็บและ CSS
st.markdown("""
    <style>
    /* 1. พื้นหลังของหน้าแอปทั้งหมด */
    .stApp {
        background-color: #e8f5e9; /* สีเขียวอ่อนแบบสบายตา */
    }
    
    /* 2. ปรับแต่งกล่อง Sidebar ด้านข้าง */
    [data-testid="stSidebar"] {
        background-color: #2e7d32;
        color: white;
    }
    
    /* 3. ทำให้ตัวหนังสือใน Sidebar เป็นสีขาว */
    [data-testid="stSidebar"] .stMarkdown p {
        color: white;
    }
    
    /* 4. ปรับแต่งปุ่มกดให้เด่นขึ้น */
    .stButton>button {
        background-color: #1b5e20 !important;
        color: white !important;
        border: 2px solid #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. ส่วนรับข้อมูล (Sidebar)
with st.sidebar:
    st.header("📌 กรอกข้อมูลนักเตะ")
    age = st.number_input("อายุ (Age)", 15, 45, 25)
    goals = st.number_input("ประตู (Goals)", 0, 100, 10)
    assists = st.number_input("แอสซิสต์ (Assists)", 0, 100, 5)
    minutes = st.number_input("นาทีที่เล่น (Minutes)", 0, 5000, 1500)
    contract = st.slider("สัญญาที่เหลือ (ปี)", 0, 5, 3)
    
    st.divider()
    st.subheader("📊 Model Info")
    st.metric("R-squared", "85.4%") # ใส่ค่าที่เราเทรนได้จริง

# 4. ส่วนการทำนาย
if st.button("ทำการพยากรณ์", key="main_prediction_btn"): # ใส่ Key เพื่อแก้ Duplicate Error
    # เตรียมข้อมูล
    features = np.array([[age, goals, assists, minutes, contract]])
    prediction = model.predict(features)[0]
    
    # แสดงผล
    st.success(f"### 💰 มูลค่าตัวที่คาดการณ์: {prediction:.2f} ล้านยูโร")
    
    # ลูกเล่น: กราฟเปรียบเทียบ
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Current Player', 'Average Top Player']
    values = [prediction, 105.0] # 105 คือค่าเฉลี่ยสมมติ
    ax.bar(labels, values, color=['#008000', '#D3D3D3'])
    ax.set_ylabel('Market Value (M€)')
    st.pyplot(fig)

    # ลูกเล่น: เทียบชั้นนักเตะ
    if prediction > 150:
        st.info("⭐ ระดับ: Super Star (Mbappe/Haaland Class)")
    elif prediction > 80:
        st.info("🔥 ระดับ: World Class (Main League Starter)")
    else:
        st.info("🏃 ระดับ: Rising Star / Experienced Player")