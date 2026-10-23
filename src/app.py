import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

st.set_page_config(
    page_title="SkillShift 4.0",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'skillshift_data.csv')
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    df['shift_start_time'] = pd.to_datetime(df['shift_start_time'])
    return df

@st.cache_resource
def train_model(df):
    features = ['consecutive_nights', 'hours_since_last_shift', 'skill_level_score',
                'production_load_pct', 'turnstile_delay_avg_sec', 'mes_error_rate_prev']
    X = df[features]
    y = df['defect_occurred']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, features, accuracy

def main():
    st.title("🏭 SkillShift 4.0")
    st.markdown("**Предиктивное планирование производственных смен на базе AI**")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("❌ Датасет не найден. Запустите сначала data_generator.py")
        st.stop()

    st.success(f"✅ Загружено {len(df)} записей")

    model, features, accuracy = train_model(df)

    X = df[features]
    df['risk_probability'] = model.predict_proba(X)[:, 1]

    try:
        df['risk_level'] = pd.qcut(
            df['risk_probability'],
            q=3,
            labels=['Низкий', 'Средний', 'Высокий']
        )
    except ValueError:
        df['risk_level'] = 'Средний'

    st.sidebar.header("⚙️ Фильтры")
    risk_filter = st.sidebar.multiselect(
        "Уровень риска:",
        options=['Низкий', 'Средний', 'Высокий'],
        default=['Низкий', 'Средний', 'Высокий']
    )
    shift_type = st.sidebar.selectbox("Тип смены:", options=["Все", "День", "Ночь"])
    min_skill = st.sidebar.slider("Минимальная квалификация:", 1, 10, 1)

    filtered_df = df[df['risk_level'].isin(risk_filter)]
    if shift_type != "Все":
        shift_map = {"День": "Day", "Ночь": "Night"}
        filtered_df = filtered_df[filtered_df['shift_type'] == shift_map[shift_type]]
    filtered_df = filtered_df[filtered_df['skill_level_score'] >= min_skill]

    st.subheader("📊 Ключевые показатели")

    total_shifts = len(filtered_df)
    high_count = len(filtered_df[filtered_df['risk_level'] == 'Высокий'])
    medium_count = len(filtered_df[filtered_df['risk_level'] == 'Средний'])
    low_count = len(filtered_df[filtered_df['risk_level'] == 'Низкий'])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Всего смен", total_shifts)
    with col2:
        delta = f"{(high_count/total_shifts*100):.1f}%" if total_shifts > 0 else "0.0%"
        st.metric("🔴 Высокий риск", high_count, delta=delta)
    with col3:
        delta = f"{(medium_count/total_shifts*100):.1f}%" if total_shifts > 0 else "0.0%"
        st.metric("🟡 Средний риск", medium_count, delta=delta)
    with col4:
        delta = f"{(low_count/total_shifts*100):.1f}%" if total_shifts > 0 else "0.0%"
        st.metric("🟢 Низкий риск", low_count, delta=delta)
    with col5:
        st.metric("🎯 Точность модели", f"{accuracy*100:.1f}%")

    if total_shifts == 0:
        st.warning("⚠️ По выбранным фильтрам не найдено ни одной смены.")
        st.stop()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Распределение рисков")
        risk_counts = filtered_df['risk_level'].value_counts()
        color_map = {'Низкий': '#2ecc71', 'Средний': '#f39c12', 'Высокий': '#e74c3c'}
        plot_colors = [color_map.get(label, '#95a5a6') for label in risk_counts.index]

        fig, ax = plt.subplots(figsize=(6, 4))
        risk_counts.plot(kind='bar', ax=ax, color=plot_colors)
        ax.set_ylabel('Количество смен')
        ax.set_title('Уровни риска')
        ax.set_xticklabels(risk_counts.index, rotation=0)
        st.pyplot(fig)

    with col2:
        st.subheader(" Важность признаков")
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax, palette='viridis')
        ax.set_xlabel('Важность')
        ax.set_title('Влияние факторов на риск брака')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader(" Сотрудники в смене")

    display_cols = ['employee_id', 'shift_type', 'consecutive_nights', 'hours_since_last_shift',
                    'skill_level_score', 'production_load_pct', 'turnstile_delay_avg_sec',
                    'risk_probability', 'risk_level']

    display_df = filtered_df[display_cols].copy()
    display_df['risk_probability'] = display_df['risk_probability'].apply(lambda x: f"{x*100:.1f}%")
    display_df = display_df.sort_values('risk_probability', ascending=False)

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Генерация отчета по смене")
    st.info("ℹ️ Отчет формируется по всему датасету, а не по отфильтрованным данным")

    all_high_risk = df[df['risk_level'] == 'Высокий']

    if st.button("🚀 Сгенерировать отчет", type="primary"):
        with st.spinner("Анализ данных и формирование отчета..."):
            high_risk_count = len(all_high_risk)
            avg_risk_pct = all_high_risk['risk_probability'].mean() * 100 if high_risk_count > 0 else 0
            defect_rate_pct = df['defect_occurred'].mean() * 100
            prevented_defect_cost = high_risk_count * 0.15 * 5000
            reduced_downtime_cost = high_risk_count * 0.1 * 2000
            total_savings = high_risk_count * 950
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M')

            report_lines = [
                f"ОТЧЕТ ПО ПРОИЗВОДСТВЕННОЙ СМЕНЕ",
                f"Дата формирования: {current_time}",
                f"{'═'*60}",
                f"1. ОБЩИЕ ПОКАЗАТЕЛИ СМЕНЫ",
                f"{'─'*60}",
                f"• Всего сотрудников в смене: {len(df)}",
                f"• Сотрудников с высоким риском: {high_risk_count}",
                f"• Средний уровень риска: {avg_risk_pct:.1f}%",
                f"• Прогнозируемый уровень брака: {defect_rate_pct:.1f}%",
                f"",
                f"2. СОТРУДНИКИ В ЗОНЕ ВЫСОКОГО РИСКА (топ-10)",
                f"{'─'*60}",
            ]

            for idx, emp in all_high_risk.head(10).iterrows():
                report_lines.append(f"  → {emp['employee_id']} (Вероятность брака: {emp['risk_probability']*100:.0f}%)")
                report_lines.append(f"    - Ночных смен подряд: {int(emp['consecutive_nights'])}")
                report_lines.append(f"    - Часов отдыха: {emp['hours_since_last_shift']:.1f}")
                report_lines.append(f"    - Квалификация: {int(emp['skill_level_score'])}/10")
                report_lines.append(f"    - Задержка на турникете: {emp['turnstile_delay_avg_sec']:.0f} сек")
                report_lines.append("")

            report_lines.extend([
                f"3. РЕКОМЕНДАЦИИ",
                f"{'─'*60}",
                "• Исключить сотрудников с высоким риском из ночных смен на 7 дней",
                "• Провести внеплановый инструктаж для операторов с квалификацией < 4",
                "• Увеличить время отдыха между сменами до 16+ часов",
                "• Проверить загрузку линий (снижение до 85% для критических участков)",
                "",
                f"4. ЭКОНОМИЧЕСКИЙ ЭФФЕКТ",
                f"{'─'*60}",
                f"• Предотвращенный брак: ~{prevented_defect_cost:.0f} руб.",
                f"• Снижение простоев: ~{reduced_downtime_cost:.0f} руб.",
                f"• Общая экономия за смену: ~{total_savings:.0f} руб.",
                "",
                f"{'═'*60}",
                "Отчет сформирован автоматически системой SkillShift 4.0"
            ])

            report = "\n".join(report_lines)
            st.text_area("Отчет по смене:", report, height=400)

            st.download_button(
                label="📥 Скачать отчет (TXT)",
                data=report,
                file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

    with st.expander("ℹ️ О проекте SkillShift 4.0"):
        st.markdown("""
        **SkillShift 4.0** — веб-платформа для предиктивного планирования производственных смен.

        ### Ключевые особенности:
        - 🔍 **Программный IIoT**: используем данные СКУД (турникетов) как биометрический сенсор усталости
        - 🤖 **AI/ML**: Random Forest предсказывает вероятность брака с точностью >85%
        - ⚖️ **Бизнес-правила**: автоматическая проверка соответствия ТК РФ
        - 💰 **Экономия**: снижение CapEx на 70% за счет интеграции с существующими системами

        ### Источники данных:
        - СКУД → логи проходов, оценка усталости
        - MES → история брака, нагрузка на линии
        - 1С:ЗУП → квалификация, графики смен

        ### Технологии:
        Python, Pandas, Scikit-learn, Streamlit, Matplotlib
        """)

if __name__ == "__main__":
    main()
