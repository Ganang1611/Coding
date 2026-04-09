import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Hybrid ARIMA-SVR DSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
# 📊 SISTEM INFORMASI PREDIKTIF  
### Estimasi Harga Bawang Merah (Hybrid ARIMA–SVR)
""")

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Pengaturan Sistem")

horizon = st.sidebar.slider("📅 Horizon Prediksi (bulan)", 1, 24, 12)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=["csv","xlsx"]
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    try:
        with open("hybrid_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model_data = load_model()

if model_data is None:
    st.error("❌ Model tidak ditemukan (hybrid_model.pkl)")
    st.stop()

svr_model = model_data["svr_model"]
arima_order = model_data["arima_order"]
feature_columns = model_data["feature_columns"]
max_lag_price = model_data["max_lag_price"]

# =====================================================
# EMPTY STATE
# =====================================================
if uploaded_file is None:
    st.info("⬅️ Silakan upload dataset pada sidebar untuk memulai analisis")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df.columns = [c.strip() for c in df.columns]

if "TAHUN BULAN" not in df.columns or "Harga" not in df.columns:
    st.error("Dataset wajib memiliki kolom: TAHUN BULAN & Harga")
    st.stop()

df["date"] = pd.to_datetime(df["TAHUN BULAN"])
df = df.sort_values("date").set_index("date")

y = df["Harga"].astype(float)

# =====================================================
# KPI DASHBOARD
# =====================================================
st.subheader("📌 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Data", len(y))
col2.metric("Harga Terakhir", f"{y.iloc[-1]:,.0f}")
col3.metric("Harga Maksimum", f"{y.max():,.0f}")
col4.metric("Harga Minimum", f"{y.min():,.0f}")

st.markdown("---")

# =====================================================
# FIT MODEL
# =====================================================
with st.spinner("⚙️ Memproses model ARIMA..."):
    arima_model = sm.tsa.SARIMAX(
        y,
        order=arima_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

arima_forecast = arima_model.forecast(steps=horizon)

# =====================================================
# HYBRID FORECAST
# =====================================================
with st.spinner("🚀 Menghitung prediksi Hybrid ARIMA-SVR..."):

    y_ext = y.copy()
    predictions = []

    for i in range(horizon):

        next_idx = y_ext.index[-1] + pd.DateOffset(months=1)

        row = {}

        for l in range(1, max_lag_price + 1):
            row[f"p_lag_{l}"] = y_ext.iloc[-l] if len(y_ext) >= l else 0

        m = next_idx.month
        row["m_sin"] = np.sin(2*np.pi*m/12)
        row["m_cos"] = np.cos(2*np.pi*m/12)

        X = pd.DataFrame([row])
        X = X.reindex(columns=feature_columns, fill_value=0)

        residual = svr_model.predict(X)[0]
        y_pred = arima_forecast.iloc[i] + residual

        predictions.append(y_pred)
        y_ext.loc[next_idx] = y_pred

# =====================================================
# HASIL FORECAST
# =====================================================
forecast_index = pd.date_range(
    y.index[-1] + pd.DateOffset(months=1),
    periods=horizon,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Forecast": predictions
}, index=forecast_index)

# =====================================================
# KPI HASIL PREDIKSI
# =====================================================
st.subheader("📊 Hasil Prediksi")

col1, col2, col3 = st.columns(3)

col1.metric("Prediksi Awal", f"{forecast_df.iloc[0,0]:,.0f}")
col2.metric("Prediksi Akhir", f"{forecast_df.iloc[-1,0]:,.0f}")
col3.metric("Rata-rata Prediksi", f"{forecast_df.mean()[0]:,.0f}")

# =====================================================
# VISUALISASI
# =====================================================
st.markdown("### 📈 Grafik Prediksi")

chart_df = pd.concat([
    y.rename("Historical"),
    forecast_df["Forecast"]
])

st.line_chart(chart_df)

# =====================================================
# TABEL
# =====================================================
st.markdown("### 📋 Detail Prediksi")
st.dataframe(forecast_df)

# =====================================================
# DOWNLOAD
# =====================================================
csv = forecast_df.to_csv().encode("utf-8")

st.download_button(
    "⬇️ Download Hasil Prediksi",
    csv,
    "forecast.csv",
    "text/csv"
)

st.markdown("---")
st.success("✅ Sistem berhasil dijalankan")