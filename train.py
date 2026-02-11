import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. โหลดข้อมูล
df = pd.read_csv('football_data.csv')

# 2. เลือก Features (5 อย่าง) และ Target
X = df[['Age', 'Goals', 'Assists', 'Minutes', 'Contract_Years']]
y = df['Market_Value (M€)']

# 3. แบ่งข้อมูลเป็น Train 80% และ Test 20% (เพื่อใช้ Evaluate)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. สร้างและเทรนโมเดล (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. การ Evaluate (ประเมินผล) - *ส่วนนี้ต้องเอาไปใส่ในสไลด์*
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"ความแม่นยำ (R-squared): {r2:.4f}")
print(f"ค่าความคลาดเคลื่อน (MSE): {mse:.4f}")

# 6. บันทึกโมเดลเป็นไฟล์ .pkl
joblib.dump(model, 'football_model.pkl')
print("บันทึกโมเดลเรียบร้อยในชื่อ 'football_model.pkl'")