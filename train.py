import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('football_data.csv')

X = df[['Age', 'Goals', 'Assists', 'Minutes', 'Contract_Years']]
y = df['Market_Value (M€)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"ความแม่นยำ (R-squared): {r2:.4f}")
print(f"ค่าความคลาดเคลื่อน (MSE): {mse:.4f}")
joblib.dump(model, 'football_model.pkl')
print("บันทึกโมเดลเรียบร้อยในชื่อ 'football_model.pkl'")