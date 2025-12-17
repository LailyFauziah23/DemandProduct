import streamlit as st
import pandas as pd
import os
from model import load_and_prepare_data, forecast, load_model

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Historical_Product_Demand.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Forecast Permintaan Produk",
    page_icon="📈",
    layout="wide"
)

# =========================
# LOAD DATA HISTORIS
# =========================
monthly_data = load_and_prepare_data(DATA_PATH)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Pengaturan Forecast")

last_date = st.sidebar.date_input(
    "Tanggal data terakhir",
    monthly_data.index.max()
)

last_demand = st.sidebar.number_input(
    "Jumlah permintaan terakhir",
    min_value=0,
    value=int(monthly_data.iloc[-1])
)

months = st.sidebar.slider(
    "Jumlah bulan yang ingin diperkirakan",
    1, 24, 12
)

# =========================
# MAIN PAGE UI
# =========================
st.title("Perkiraan Permintaan Produk")
st.write(
    "Aplikasi ini memperkirakan **jumlah permintaan produk** di bulan-bulan berikutnya "
    "berdasarkan data historis."
)

st.subheader("Data Historis Terakhir")
st.write(monthly_data.tail())

# =========================
# FORECAST
# =========================
if st.button("Jalankan Perkiraan"):
    # Tambahkan nilai terakhir user
    new_point = pd.Series([last_demand], index=[pd.to_datetime(last_date)])
    updated_data = pd.concat([monthly_data, new_point])
    updated_data = updated_data[~updated_data.index.duplicated(keep='last')]

    # Load model SARIMA
    model = load_model("models/sarima_model.pkl") 
    mean, ci = forecast(model, steps=months)

    # =========================
    # TABEL HASIL FORECAST
    # =========================
    hasil = pd.DataFrame({
        "Perkiraan Permintaan": mean.astype(int),
        "Perkiraan Terendah": ci.iloc[:, 0].astype(int),
        "Perkiraan Tertinggi": ci.iloc[:, 1].astype(int)
    })

    st.subheader("Hasil Perkiraan")
    st.dataframe(hasil)

    st.caption(
        "Perkiraan terendah dan tertinggi menunjukkan rentang kemungkinan permintaan yang dapat terjadi."
    )

    # =========================
    # GRAFIK HISTORIS + FORECAST
    # =========================
    combined = pd.concat([
        updated_data.to_frame("Data Historis"),
        mean.to_frame("Perkiraan Permintaan")
    ])

    st.subheader("Grafik Perkiraan Permintaan")
    st.line_chart(combined)
