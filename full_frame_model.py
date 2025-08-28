# full_frame_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

print("🔧 Загружаем данные...")
df = pd.read_csv('data.csv')

# --- Преобразуем дату ---
df['date'] = pd.to_datetime(df['date'])
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# --- Признаки: добавляем длины и щиты ---
features = [
    'base_area',
    'height',
    'num_floors',
    'total_post_length_m',       # суммарная длина стоек
    'total_beam_length_m',       # суммарная длина перемычек
    'num_panels',                # количество щитов
    'shape_type',
    'days_since_start'
]

X = df[features]
y = df['target_price']

print(f"✅ Признаков: {len(features)}, Проектов: {len(df)}")

# --- Разделение ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Модель ---
print("🚀 Обучаем XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# --- Прогноз ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = model.score(X_test, y_test)

print(f"\n📊 Результаты на тесте:")
print(f"Средняя ошибка: {mae:,.0f} руб.")
print(f"Ошибка в % (MAPE): {mape:.1f}%")
print(f"Точность (R²): {r2:.3f}")

# --- Пример прогноза ---
print(f"\n🧩 Пример прогноза:")
new = [[123.0, 5.95, 4, 284.0, 440.0, 59, 2, 500]]
price = model.predict(new)[0]
print(f"🎯 Прогноз: {price:,.0f} руб.")

# --- Сохранение ---
model.save_model('full_frame_model.json')
print("✅ Модель сохранена как 'full_frame_model.json'")