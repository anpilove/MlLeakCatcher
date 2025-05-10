import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    st.set_page_config(page_title="MlLeakCatcher: Анализ утечек данных", layout="wide")
    st.title("MlLeakCatcher: Инструмент для анализа утечек данных")

    st.markdown("""
    Добро пожаловать в **MlLeakCatcher** — мощный инструмент для обнаружения утечек данных и оценки их воздействия на качество моделей.
    Загрузите ваш датасет и выберите метод анализа, чтобы проверить возможные утечки данных в вашем наборе.
    """, unsafe_allow_html=True)

    st.sidebar.header("Загрузите ваши данные")
    uploaded_file = st.sidebar.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("**Загруженные данные**:")
        st.dataframe(df.head())

        st.sidebar.subheader("Выберите метод анализа утечек")
        analysis_option = st.sidebar.selectbox(
            "Метод анализа утечек данных:",
            [
                "Оценка утечки по одному признаку",
                "Обнаружение идентификаторов и уникальных признаков",
                "Анализ связи признака с целевой переменной",
                "Сравнение важности всех признаков в простой и сложной модели"
            ]
        )


    else:
        st.write("Пожалуйста, загрузите файл с данными для начала анализа.")


if __name__ == "__main__":
    main()
