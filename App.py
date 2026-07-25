import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import requests
import time
import io

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO DE SESIÓN
# ==============================================================================
st.set_page_config(page_title="Dashboard Trading ML", layout="wide", page_icon="📈")

if "ultimo_evento_alertado" not in st.session_state:
    st.session_state.ultimo_evento_alertado = None
if "ultimo_rechazo_alertado" not in st.session_state:
    st.session_state.ultimo_rechazo_alertado = None

# ==============================================================================
# CONFIGURACIÓN DE TELEGRAM
# ==============================================================================
TELEGRAM_BOT_TOKEN = "8563258258:AAEeqztdaW5zfTWHywv-UfS3w8-SpHwgDVg"
TELEGRAM_CHAT_ID = "-1003571551698"
TIEMPO_ESPERA_SEGUNDOS = 300  # 5 minutos

ny_tz = pytz.timezone('America/New_York')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.sidebar.error(f"Error Telegram: {e}")

def send_telegram_photo(fig):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#0e1117')
        buf.seek(0)
        payload = {'chat_id': TELEGRAM_CHAT_ID}
        files = {'photo': buf}
        requests.post(url, data=payload, files=files)
    except Exception as e:
        st.sidebar.error(f"Error enviando foto a Telegram: {e}")

# ==============================================================================
# INTERFAZ DE USUARIO (BARRA LATERAL)
# ==============================================================================
st.sidebar.title("⚙️ Configuración")
ticker_seleccionado = st.sidebar.selectbox(
    "Selecciona el Activo", 
    options=["^NDX", "GC=F"], 
    format_func=lambda x: "Nasdaq 100 (^NDX)" if x == "^NDX" else "Oro (GC=F)"
)

st.sidebar.markdown("---")
st.sidebar.info("El sistema se actualiza automáticamente cada 5 minutos.")

# ==============================================================================
# MOTOR PRINCIPAL
# ==============================================================================
now_ny = datetime.now(ny_tz)
st.title(f"📊 Dashboard en vivo: {ticker_seleccionado}")
st.markdown(f"**Última actualización:** {now_ny.strftime('%Y-%m-%d %H:%M:%S')} EST")

with st.spinner("Descargando datos y entrenando modelos..."):
    try:
        # ----------------------------------------------------------------------
        # 1. DATOS DIARIOS (GARCH)
        # ----------------------------------------------------------------------
        data_1d = yf.download(ticker_seleccionado, period="3y", interval="1d", progress=False)
        prices_1d = data_1d['Close'][ticker_seleccionado] if isinstance(data_1d.columns, pd.MultiIndex) else data_1d['Close']
        prices_1d = prices_1d.ffill().dropna().squeeze()
        returns_1d = 100 * prices_1d.pct_change().dropna()

        model = arch_model(returns_1d, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
        res = model.fit(update_freq=5, disp='off')
        volatility_1d = res.conditional_volatility

        last_price_1d = float(prices_1d.iloc[-1])
        forecasts = res.forecast(horizon=5, method='simulation', simulations=10000)
        sim_paths = forecasts.simulations.values[-1, :, :]
        projected_prices_1d = last_price_1d * np.exp(np.cumsum(sim_paths / 100, axis=1))

        percentiles_list = [0.05, 0.25, 0.50, 0.75, 0.95]
        fechas_forecast = [prices_1d.index[-1] + timedelta(days=i) for i in range(1, 6)]
        df_forecast = pd.DataFrame(index=fechas_forecast)

        for p in percentiles_list:
            df_forecast[f'P{int(p*100)}'] = np.percentile(projected_prices_1d, p * 100, axis=0)

        # ----------------------------------------------------------------------
        # 2. DATOS DE 5 MINUTOS (ESTRUCTURA Y ML)
        # ----------------------------------------------------------------------
        data_5m = yf.download(ticker_seleccionado, period="60d", interval="5m", progress=False)

        if isinstance(data_5m.columns, pd.MultiIndex):
            prices_5m = data_5m['Close'][ticker_seleccionado]
            highs_5m = data_5m['High'][ticker_seleccionado]
            lows_5m = data_5m['Low'][ticker_seleccionado]
        else:
            prices_5m = data_5m['Close']
            highs_5m = data_5m['High']
            lows_5m = data_5m['Low']

        prices_5m = prices_5m.ffill().dropna().squeeze()
        highs_5m = highs_5m.ffill().dropna().squeeze()
        lows_5m = lows_5m.ffill().dropna().squeeze()

        vol_df = volatility_1d.to_frame(name='vol')
        vol_df.index = vol_df.index.strftime('%Y-%m-%d')
        str_dates_5m = prices_5m.index.tz_convert(ny_tz).strftime('%Y-%m-%d')
        
        volatility_5m = pd.Series(str_dates_5m.map(vol_df['vol']).values, index=prices_5m.index)
        volatility_5m = volatility_5m.ffill().bfill()

        # ----------------------------------------------------------------------
        # 3. DETECCIÓN Y ML
        # ----------------------------------------------------------------------
        def detect_swings_and_events(prices_series, high_series, low_series, vol_series, window=5):
            arr_p, arr_h, arr_l = prices_series.values, high_series.values, low_series.values
            arr_v = vol_series.values
            idx = prices_series.index
            events, swing_highs, swing_lows = [], [], []

            for i in range(window, len(arr_p) - window):
                if arr_h[i] == np.max(arr_h[i-window : i+window+1]):
                    swing_highs.append((i, idx[i], arr_h[i]))
                if arr_l[i] == np.min(arr_l[i-window : i+window+1]):
                    swing_lows.append((i, idx[i], arr_l[i]))

            current_trend = 1
            X_features, y_ext_reg, y_reach_cls = [], [], []

            for i in range(window*2, len(arr_p)):
                close_i, high_i, low_i, date_i, vol_i = arr_p[i], arr_h[i], arr_l[i], idx[i], arr_v[i]
                prev_shs = [sh for sh in swing_highs if sh[0] < i]
                prev_sls = [sl for sl in swing_lows if sl[0] < i]

                if not prev_shs or not prev_sls: continue
                recent_sh, recent_sl = prev_shs[-1][2], prev_sls[-1][2]
                event_type = None

                if close_i > recent_sh and arr_p[i-1] <= recent_sh:
                    event_type = "BOS Alcista" if current_trend == 1 else "CHOCH Alcista"
                    current_trend = 1
                    break_level = recent_sh
                    leg_size = max(1e-5, recent_sh - recent_sl)
                    future_max = np.max(arr_h[i:min(i+15, len(arr_h))])
                    extension_ratio = (future_max - break_level) / leg_size
                    X_features.append([leg_size / close_i, vol_i, (close_i - arr_p[i-5])/arr_p[i-5]])
                    y_ext_reg.append(extension_ratio)
                    y_reach_cls.append(1 if extension_ratio >= 1.272 else 0)

                elif close_i < recent_sl and arr_p[i-1] >= recent_sl:
                    event_type = "BOS Bajista" if current_trend == -1 else "CHOCH Bajista"
                    current_trend = -1
                    break_level = recent_sl
                    leg_size = max(1e-5, recent_sh - recent_sl)
                    future_min = np.min(arr_l[i:min(i+15, len(arr_l))])
                    extension_ratio = (break_level - future_min) / leg_size
                    X_features.append([leg_size / close_i, vol_i, (close_i - arr_p[i-5])/arr_p[i-5]])
                    y_ext_reg.append(extension_ratio)
                    y_reach_cls.append(1 if extension_ratio >= 1.272 else 0)

                if event_type:
                    events.append({
                        'index': i, 'date': date_i, 'type': event_type, 'price': close_i,
                        'break_level': break_level, 'leg_size': leg_size, 'vol': vol_i,
                        'ret_5d': (close_i - arr_p[i-5])/arr_p[i-5]
                    })
            return events, np.array(X_features), np.array(y_ext_reg), np.array(y_reach_cls)

        events, X, y_reg, y_cls = detect_swings_and_events(prices_5m, highs_5m, lows_5m, volatility_5m)

        ml_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        ml_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        if len(X) > 10:
            ml_regressor.fit(X, y_reg)
            ml_classifier.fit(X, y_cls)

        # ----------------------------------------------------------------------
        # 4. RECHAZOS GARCH Y LOGS
        # ----------------------------------------------------------------------
        p5_val = df_forecast['P5'].iloc[0]
        p25_val = df_forecast['P25'].iloc[0]
        p50_val = df_forecast['P50'].iloc[0]
        p75_val = df_forecast['P75'].iloc[0]
        p95_val = df_forecast['P95'].iloc[0]

        garch_rejection = None
        last_low = lows_5m.iloc[-1]
        last_high = highs_5m.iloc[-1]
        last_close = prices_5m.iloc[-1]

        if last_high >= p95_val and last_close < p95_val:
            garch_rejection = f"Rechazo Resistencia Extrema GARCH (P95: {p95_val:.2f})"
        elif last_low <= p5_val and last_close > p5_val:
            garch_rejection = f"Rechazo Soporte Extremo GARCH (P5: {p5_val:.2f})"
        elif last_high >= p75_val and last_close < p75_val:
            garch_rejection = f"Rechazo Resistencia GARCH (P75: {p75_val:.2f})"
        elif last_low <= p25_val and last_close > p25_val:
            garch_rejection = f"Rechazo Soporte GARCH (P25: {p25_val:.2f})"
        elif last_high >= p50_val and last_close < p50_val:
            garch_rejection = f"Rechazo Nivel Medio GARCH (Resistencia P50: {p50_val:.2f})"
        elif last_low <= p50_val and last_close > p50_val:
            garch_rejection = f"Rechazo Nivel Medio GARCH (Soporte P50: {p50_val:.2f})"

        analysis_logs = []
        for ev in events[-7:]:
            feat = np.array([[ev['leg_size'] / ev['price'], ev['vol'], ev['ret_5d']]])
            if len(X) > 10:
                pred_ext_mult = float(ml_regressor.predict(feat)[0])
                pred_prob = float(ml_classifier.predict_proba(feat)[0][1]) * 100
            else:
                pred_ext_mult = 1.272
                pred_prob = 65.0

            target_price = ev['price'] + (ev['leg_size'] * pred_ext_mult) if "Alcista" in ev['type'] else ev['price'] - (ev['leg_size'] * pred_ext_mult)

            event_dt = pd.to_datetime(ev['date'])
            if event_dt.tzinfo is None:
                event_dt = pytz.utc.localize(event_dt).astimezone(ny_tz)
            else:
                event_dt = event_dt.astimezone(ny_tz)

            log_time = event_dt.strftime('%Y-%m-%d %H:%M:%S EST')
            
            analysis_logs.append({
                'id': f"{ticker_seleccionado}_{event_dt.strftime('%Y%m%d_%H%M%S')}_{ev['type']}",
                'timestamp_ny': log_time,
                'event': ev['type'],
                'price': ev['price'],
                'target_price': target_price,
                'prob': pred_prob,
                'detail': f"🎯 {ev['type']} detectado en {ev['price']:.2f}.\n📈 Objetivo ML: {target_price:.2f}\n⚡ Probabilidad: {pred_prob:.1f}%"
            })

        if garch_rejection:
            analysis_logs[-1]['detail'] += f"\n\n🚨 *ALERTA:* {garch_rejection}"

        # ----------------------------------------------------------------------
        # 5. RENDERIZADO VISUAL DEL DASHBOARD WEB
        # ----------------------------------------------------------------------
        col1, col2 = st.columns([2, 1])

        with col1:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(prices_5m.tail(150), color='#00d1ff', label=f'{ticker_seleccionado} (5m Close)', linewidth=1.5)

            for ev in events[-10:]:
                if ev['date'] in prices_5m.tail(150).index:
                    color = '#00ff88' if 'Alcista' in ev['type'] else '#ff4b4b'
                    marker = '^' if 'Alcista' in ev['type'] else 'v'
                    ax.plot(ev['date'], ev['price'], marker=marker, color=color, markersize=10)
                    ax.annotate(ev['type'], (ev['date'], ev['price']), textcoords="offset points",
                                xytext=(0, 10 if 'Alcista' in ev['type'] else -15),
                                ha='center', color=color, fontsize=8, weight='bold')

            ax.axhline(p95_val, color='#ff6b6b', linestyle='--', alpha=0.6, label=f'GARCH Res (P95: {p95_val:.0f})')
            ax.axhline(p75_val, color='#ff9f43', linestyle=':', alpha=0.5, label=f'GARCH Res (P75: {p75_val:.0f})')
            ax.axhline(p50_val, color='#feca57', linestyle='-.', alpha=0.6, label=f'GARCH Med (P50: {p50_val:.0f})')
            ax.axhline(p25_val, color='#1dd1a1', linestyle=':', alpha=0.5, label=f'GARCH Sop (P25: {p25_val:.0f})')
            ax.axhline(p5_val, color='#10ac84', linestyle='--', alpha=0.6, label=f'GARCH Sop (P5: {p5_val:.0f})')

            ax.set_title(f"Estructura 5M + Niveles GARCH 1D", fontsize=12, pad=15)
            ax.grid(True, alpha=0.08)
            ax.legend(loc='upper left', fontsize=8)
            fig.patch.set_facecolor('#0e1117') 
            ax.set_facecolor('#0e1117')
            
            st.pyplot(fig)

        with col2:
            st.subheader("Pronóstico GARCH (5 Días)")
            st.dataframe(df_forecast.round(2), use_container_width=True)

        st.markdown("---")
        st.subheader("📜 Registro de Eventos Estructurales")
        
        for log in reversed(analysis_logs):
            with st.expander(f"{log['timestamp_ny']} | {log['event']} | Precio: {log['price']:.2f}"):
                st.markdown(log['detail'])

        # ----------------------------------------------------------------------
        # 6. LÓGICA DE ALERTAS TELEGRAM
        # ----------------------------------------------------------------------
        ultimo_evento_detectado = analysis_logs[-1]
        enviar_foto = False

        if ultimo_evento_detectado['id'] != st.session_state.ultimo_evento_alertado:
            msg = f"📊 *Nueva Estructura 5M ({ticker_seleccionado})*\n\n{ultimo_evento_detectado['detail']}\n\n🕒 {ultimo_evento_detectado['timestamp_ny']}"
            send_telegram_message(msg)
            st.session_state.ultimo_evento_alertado = ultimo_evento_detectado['id']
            enviar_foto = True

        if garch_rejection and garch_rejection != st.session_state.ultimo_rechazo_alertado:
            msg_rechazo = f"⚠️ *Rechazo Nivel GARCH ({ticker_seleccionado})*\n\nSe ha detectado: {garch_rejection}\nNivel actual: {prices_5m.iloc[-1]:.2f}\n\n🕒 {now_ny.strftime('%H:%M:%S')} EST"
            send_telegram_message(msg_rechazo)
            st.session_state.ultimo_rechazo_alertado = garch_rejection
            enviar_foto = True

        if enviar_foto:
            send_telegram_photo(fig)

    except Exception as e:
        st.error(f"Error procesando datos: {e}")

# ==============================================================================
# BUCLE DE ACTUALIZACIÓN (AUTO-REFRESH)
# ==============================================================================
time.sleep(TIEMPO_ESPERA_SEGUNDOS)
st.rerun()
