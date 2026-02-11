import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
    <style>
    /* 1. เปลี่ยนพื้นหลังทั้งหมดเป็นโทนเข้มแบบสปอร์ต */
    .stApp {
        background: linear-gradient(135deg, #111111 0%, #0a2e0a 100%);
        color: #ffffff;
    }

    /* 2. ปรับแต่ง Sidebar ให้ดูหรูขึ้น */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7);
        border-right: 2px solid #2e7d32;
    }

    /* 3. ปรับแต่งปุ่มกด (Button) ให้เป็นสีเขียวนีออน */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.4);
    }
    
    .stButton>button:hover {
        background-color: #388e3c;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.6);
    }

    /* 4. ปรับแต่ง Input Box */
    .stNumberInput input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #2e7d32 !important;
    }

    /* 5. ปรับแต่ง Metric (ตัวเลข R-squared) */
    [data-testid="stMetricValue"] {
        color: #4caf50 !important;
        font-size: 2rem;
    }

    /* 6. การ์ดแสดงผลลัพธ์ (Success Message) */
    .stAlert {
        background-color: rgba(46, 125, 50, 0.2);
        border: 1px solid #4caf50;
        color: #ffffff;
        border-radius: 15px;
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