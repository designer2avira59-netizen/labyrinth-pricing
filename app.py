# app.py
import streamlit as st
import xgboost as xgb
import pandas as pd

# --- Заголовок ---
st.title("⚡ Быстрая оценка стоимости каркаса")
st.write("Введите параметры — получите прогноз за секунду.")

# --- Загружаем модель ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('fast_frame_model.json')
    return model

model = load_model()

# --- Поля ввода ---
base_area = st.number_input(
    "Площадь основания (м²)",
    min_value=5.0,
    max_value=400.0,
    value=100.0,
    step=1.0
)

height = st.number_input(
    "Высота каркаса (м)",
    min_value=2.0,
    max_value=10.0,
    value=4.0,
    step=0.1
)

num_floors = st.number_input(
    "Количество этажей",
    min_value=1,
    max_value=6,
    value=3,
    step=1
)

shape_type = st.selectbox(
    "Форма каркаса",
    options=[0, 1, 2, 4, 5],
    format_func=lambda x: {
        0: "0 - Куб",
        1: "1 - O-образный (с вырезом)",
        2: "2 - П-образный",
        4: "4 - Г-образный",
        5: "5 - Линия"
    }[x]
)

# --- Кнопка расчёта ---
if st.button("🧮 Рассчитать стоимость"):
    # Считаем days_since_start (дней с 2025-07-01)
    from datetime import datetime
    start_date = datetime(2025, 7, 1)
    today = datetime.today()
    days = (today - start_date).days

    # Подготовка данных
    input_data = [[base_area, height, num_floors, shape_type, days]]

    # Прогноз
    predicted_price = model.predict(input_data)[0]

    # Вывод
    st.success(f"🎯 **Прогноз стоимости каркаса: {predicted_price:,.0f} руб.**")
    st.info(f"📅 Дней с 01.07.2025: {days}")

# --- Подсказка ---
with st.expander("💡 Как использовать"):
    st.markdown("""
    1. Введите площадь, высоту, этажность и форму.
    2. Нажмите «Рассчитать».
    3. Результат появится мгновенно.
    
    ⚠️ Модель обучена на реальных данных и даёт оценку с точностью ~10%.
    """)