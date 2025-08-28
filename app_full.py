# app_full.py
import streamlit as st
import xgboost as xgb

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('full_frame_model.json')
    return model

model = load_model()

# --- Интерфейс ---
st.title("🧱 Полная оценка стоимости каркаса")
st.write("Введите точные параметры — получите максимально точный прогноз.")

# --- Поля ввода ---
base_area = st.number_input("Площадь основания (м²)", 10.0, 300.0, 123.0)
height = st.number_input("Высота (м)", 2.0, 10.0, 5.95)
num_floors = st.number_input("Этажей", 1, 6, 4)
total_post_length = st.number_input("Суммарная длина стоек (м)", 10.0, 1000.0, 284.0)
total_beam_length = st.number_input("Суммарная длина перемычек (м)", 10.0, 1000.0, 440.0)
num_panels = st.number_input("Количество щитов", 1, 100, 59)
shape_type = st.selectbox(
    "Форма каркаса",
    options=[0, 1, 2, 4, 5],
    format_func=lambda x: {
        0: "0 - Куб",
        1: "1 - O-образный",
        2: "2 - П-образный",
        4: "4 - Г-образный",
        5: "5 - Линия"
    }[x]
)

# --- Кнопка расчёта ---
if st.button("🧮 Рассчитать стоимость"):
    # Считаем days_since_start
    from datetime import datetime
    start_date = datetime(2025, 7, 1)
    today = datetime.today()
    days = (today - start_date).days

    # Подготовка данных
    input_data = [[base_area, height, num_floors, total_post_length, total_beam_length, num_panels, shape_type, days]]

    # Прогноз
    predicted_price = model.predict(input_data)[0]

    # Вывод
    st.success(f"🎯 **Прогноз стоимости каркаса: {predicted_price:,.0f} руб.**")
    st.info(f"📅 Дней с 01.07.2025: {days}")