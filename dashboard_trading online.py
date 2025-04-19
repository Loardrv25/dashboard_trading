import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from prophet import Prophet
import os

import streamlit as st

API_KEY = st.secrets["ALPHA_VANTAGE_KEY"]


# Función para obtener datos
@st.cache_data
def obtener_datos_activo(symbol):
    ts = TimeSeries(API_KEY, output_format='pandas')
    datos, _ = ts.get_daily(symbol=symbol, outputsize='full')
    datos.sort_index(inplace=True)
    return datos['4. close']

# Indicadores técnicos
def calcular_indicadores(df):
    df = df.to_frame(name='Precio')

    # SMA
    df['SMA_20'] = df['Precio'].rolling(window=20).mean()
    df['SMA_50'] = df['Precio'].rolling(window=50).mean()

    # RSI
    delta = df['Precio'].diff(1)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=14).mean()
    ma_down = down.rolling(window=14).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    df['RSI'] = rsi

    # Bollinger Bands
    df['Bollinger_Mid'] = df['Precio'].rolling(window=20).mean()
    df['Bollinger_Up'] = df['Bollinger_Mid'] + 2 * df['Precio'].rolling(window=20).std()
    df['Bollinger_Down'] = df['Bollinger_Mid'] - 2 * df['Precio'].rolling(window=20).std()

    return df

# Recomendación basada en SMA
def recomendacion_sma(df):
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
        return "Compra", "#4CAF50"
    elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]:
        return "Venta", "#F44336"
    else:
        return "Mantener", "#808080"

# Recomendación basada en RSI
def recomendacion_rsi(df):
    rsi = df['RSI'].iloc[-1]
    if rsi < 30:
        return "Compra", "#4CAF50"
    elif rsi > 70:
        return "Venta", "#F44336"
    else:
        return "Mantener", "#808080"

# Dashboard Streamlit
st.title("📊 Dashboard Análisis Técnico")

activo = st.selectbox("Selecciona el activo", ["BTCUSD", "ETHUSD", "AAPL", "NVDA"])
precios = obtener_datos_activo(activo)
indicadores_df = calcular_indicadores(precios)

# Filtrar últimos 2 años
fecha_limite = indicadores_df.index[-1] - pd.DateOffset(years=2)
indicadores_df = indicadores_df[indicadores_df.index >= fecha_limite]

# Último precio
ultimo_precio = precios.iloc[-1]
st.markdown(f"### 📌 Precio actual de {activo}: **{ultimo_precio:.2f} USD**")

# Gráfico principal con SMA y Bollinger Bands
fig = go.Figure()
fig.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['Precio'], name='Precio', line=dict(width=2)))
fig.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['SMA_20'], name='SMA 20', line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['SMA_50'], name='SMA 50', line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['Bollinger_Up'], name='Bollinger Superior', line=dict(width=1, dash='dot')))
fig.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['Bollinger_Down'], name='Bollinger Inferior', line=dict(width=1, dash='dot')))

fig.update_layout(title=f"Análisis Técnico de {activo}", xaxis_title="Fecha", yaxis_title="Precio (USD)")
st.plotly_chart(fig)

# Barra recomendaciones SMA y RSI
rec_sma, color_sma = recomendacion_sma(indicadores_df)
rec_rsi, color_rsi = recomendacion_rsi(indicadores_df)
st.markdown(f"### 🔖 Recomendación SMA: <span style='color:{color_sma}'>{rec_sma}</span>", unsafe_allow_html=True)
st.markdown(f"### 🔖 Recomendación RSI: <span style='color:{color_rsi}'>{rec_rsi}</span>", unsafe_allow_html=True)

# Gráfico RSI
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=indicadores_df.index, y=indicadores_df['RSI'], name='RSI', line=dict(color='orange')))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
fig_rsi.update_layout(title="Índice RSI", yaxis_title="RSI", xaxis_title="Fecha")
st.plotly_chart(fig_rsi)

# Predicción Prophet (sin cambios)
st.markdown("## 🔮 Predicción de precios (30 días)")
df_prophet = indicadores_df.reset_index().rename(columns={'date':'ds', 'Precio':'y'})
modelo = Prophet(daily_seasonality=True)
modelo.fit(df_prophet)

futuro = modelo.make_future_dataframe(periods=30)
predicciones = modelo.predict(futuro)

fig_prediccion = modelo.plot(predicciones)
st.pyplot(fig_prediccion)

# Recomendación Prophet
precio_predicho = predicciones['yhat'].iloc[-1]
precio_actual = df_prophet['y'].iloc[-1]
if precio_predicho > precio_actual * 1.02:
    rec_prophet, color_prophet = "Compra", "#4CAF50"
elif precio_predicho < precio_actual * 0.98:
    rec_prophet, color_prophet = "Venta", "#F44336"
else:
    rec_prophet, color_prophet = "Mantener", "#808080"
st.markdown(f"### 🔖 Recomendación Prophet: <span style='color:{color_prophet}'>{rec_prophet}</span>", unsafe_allow_html=True)
