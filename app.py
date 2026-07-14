import streamlit as st

from ui import apply_style


st.set_page_config(
    page_title="Forecast Rawat Jalan",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_style()

pages = {
    "Ringkasan": [
        st.Page("pages/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
    ],
    "Analisis": [
        st.Page("pages/forecasting_page.py", title="Forecasting", icon=":material/query_stats:"),
        st.Page("pages/evaluation_page.py", title="Evaluasi Model", icon=":material/fact_check:"),
        st.Page("pages/history_page.py", title="Data Historis", icon=":material/calendar_month:"),
    ],
    "Sistem": [
        st.Page("pages/model_info.py", title="Informasi Model", icon=":material/info:"),
    ],
}

with st.sidebar:
    st.markdown("### Forecast Rawat Jalan")
    st.caption("Perencanaan kapasitas poliklinik")

navigation = st.navigation(pages)
navigation.run()
