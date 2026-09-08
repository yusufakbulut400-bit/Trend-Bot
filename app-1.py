"""
TREND FOLLOWING ANALİZ BOTU — WEB UYGULAMASI
==============================================
Streamlit ile hazırlanmış, tarayıcıdan çalışan versiyon.
Kullanıcılar hiçbir kurulum yapmadan, sadece link üzerinden
sembol girip trend analizini görebilir.

Yerelde çalıştırmak için:
    pip install streamlit pandas numpy yfinance
    streamlit run app.py

Herkese açık link için Streamlit Community Cloud'a deploy edilir
(README_DEPLOY.md dosyasına bak).
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    yf = None


# ============================================================
# İNDİKATÖR FONKSİYONLARI (trend_following_bot.py ile aynı mantık)
# ============================================================

def sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def roc(series, window=10):
    return (series / series.shift(window) - 1) * 100


def donchian_channel(df, window=20):
    upper = df["High"].rolling(window).max()
    lower = df["Low"].rolling(window).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


def linear_regression_channel(series, window=50):
    slopes = pd.Series(index=series.index, dtype=float)
    r_squared = pd.Series(index=series.index, dtype=float)
    x = np.arange(window)
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1: i + 1].values
        if np.isnan(y).any():
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        slopes.iloc[i] = slope
        r_squared.iloc[i] = r2
    norm_slope = slopes / series * 100
    return norm_slope, r_squared


@st.cache_data(ttl=300)
def fetch_data(symbol, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"'{symbol}' için veri bulunamadı.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    return df.dropna()


def analyze_trend(df, symbol=""):
    close = df["Close"]
    df = df.copy()
    df["SMA20"] = sma(close, 20)
    df["SMA50"] = sma(close, 50)
    df["SMA200"] = sma(close, 200)
    df["RSI14"] = rsi(close, 14)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(close)
    df["ROC10"] = roc(close, 10)
    df["Donchian_Upper"], df["Donchian_Mid"], df["Donchian_Lower"] = donchian_channel(df, 20)
    df["LR_Slope_%"], df["LR_R2"] = linear_regression_channel(close, window=50)

    last = df.iloc[-1]
    score, max_score = 0.0, 0.0

    ma_weight = 35
    if not np.isnan(last["SMA200"]):
        if last["Close"] > last["SMA50"] > last["SMA200"]:
            ma_score, ma_signal = 1.0, "Güçlü Yükseliş (Fiyat > SMA50 > SMA200)"
        elif last["Close"] < last["SMA50"] < last["SMA200"]:
            ma_score, ma_signal = -1.0, "Güçlü Düşüş (Fiyat < SMA50 < SMA200)"
        elif last["Close"] > last["SMA50"]:
            ma_score, ma_signal = 0.4, "Zayıf Yükseliş"
        else:
            ma_score, ma_signal = -0.4, "Zayıf Düşüş"
    else:
        if last["Close"] > last["SMA20"]:
            ma_score, ma_signal = 0.5, "Kısa Vadeli Yükseliş (yetersiz veri)"
        else:
            ma_score, ma_signal = -0.5, "Kısa Vadeli Düşüş (yetersiz veri)"
    score += ma_score * ma_weight
    max_score += ma_weight

    mom_weight = 30
    macd_bull = last["MACD"] > last["MACD_signal"]
    roc_bull = last["ROC10"] > 0
    if macd_bull and roc_bull:
        mom_score, momentum_signal = 1.0, "Pozitif (MACD ve ROC yukarı)"
    elif (not macd_bull) and (not roc_bull):
        mom_score, momentum_signal = -1.0, "Negatif (MACD ve ROC aşağı)"
    elif macd_bull:
        mom_score, momentum_signal = 0.3, "Karışık (MACD yukarı, ROC aşağı)"
    else:
        mom_score, momentum_signal = -0.3, "Karışık (MACD aşağı, ROC yukarı)"
    score += mom_score * mom_weight
    max_score += mom_weight

    channel_weight = 35
    slope = last["LR_Slope_%"]
    r2 = last["LR_R2"] if not np.isnan(last["LR_R2"]) else 0.0
    if np.isnan(slope):
        channel_score, channel_direction = 0, "Belirsiz (yetersiz veri)"
    elif slope > 0.05:
        channel_score, channel_direction = min(1.0, r2 + 0.3), "Yükselen Kanal"
    elif slope < -0.05:
        channel_score, channel_direction = -min(1.0, r2 + 0.3), "Düşen Kanal"
    else:
        channel_score, channel_direction = 0, "Yatay Kanal"
    score += channel_score * channel_weight
    max_score += channel_weight

    normalized = (score / max_score) * 100 if max_score else 0
    confidence = min(100, abs(normalized) + (r2 * 15))

    if normalized > 15:
        trend = "YÜKSELİŞ TRENDİ"
    elif normalized < -15:
        trend = "DÜŞÜŞ TRENDİ"
    else:
        trend = "YATAY / BELİRSİZ"

    return df, {
        "trend": trend,
        "confidence": confidence,
        "channel_direction": channel_direction,
        "channel_r2": r2,
        "ma_signal": ma_signal,
        "momentum_signal": momentum_signal,
        "rsi_value": last["RSI14"] if not np.isnan(last["RSI14"]) else 0.0,
        "last_price": last["Close"],
    }


# ============================================================
# ARAYÜZ (STREAMLIT)
# ============================================================

st.set_page_config(page_title="Trend Following Analiz Botu", page_icon="📈", layout="wide")

st.title("📈 Trend Following Analiz Botu")
st.caption("Hareketli ortalama + momentum + kanal analiziyle trend yönü tespiti")
st.markdown("**Geliştirici:** Yusuf İslam Akbulut")

with st.sidebar:
    st.header("Ayarlar")
    symbol = st.text_input("Sembol", value="BTC-USD",
                            help="Örn: BTC-USD, THYAO.IS, AAPL, EURUSD=X")
    period = st.selectbox("Zaman Aralığı", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Mum Periyodu", ["1d", "1h", "1wk"], index=0)
    run = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

st.warning(
    "⚠️ Bu araç yalnızca teknik analiz amaçlıdır, yatırım tavsiyesi değildir. "
    "Yatırım kararlarının sorumluluğu tamamen kullanıcıya aittir.",
    icon="⚠️",
)

if run:
    if yf is None:
        st.error("yfinance kurulu değil.")
    else:
        try:
            with st.spinner(f"{symbol} için veri çekiliyor ve analiz ediliyor..."):
                raw_df = fetch_data(symbol, period=period, interval=interval)
                df, result = analyze_trend(raw_df, symbol=symbol)

            trend_color = {
                "YÜKSELİŞ TRENDİ": "green",
                "DÜŞÜŞ TRENDİ": "red",
                "YATAY / BELİRSİZ": "orange",
            }[result["trend"]]

            col1, col2, col3 = st.columns(3)
            col1.metric("Son Fiyat", f"{result['last_price']:.4f}")
            col2.markdown(
                f"### Trend: :{trend_color}[{result['trend']}]"
            )
            col3.metric("Güven Skoru", f"{result['confidence']:.0f} / 100")

            st.divider()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Kanal Yönü**\n\n{result['channel_direction']}  \n(R²={result['channel_r2']:.2f})")
            c2.markdown(f"**MA Sinyali**\n\n{result['ma_signal']}")
            c3.markdown(f"**Momentum**\n\n{result['momentum_signal']}  \nRSI(14): {result['rsi_value']:.1f}")

            st.divider()
            st.subheader("Fiyat Grafiği")

            chart_df = df[["Close", "SMA50", "SMA200", "Donchian_Upper", "Donchian_Lower"]]
            st.line_chart(chart_df)

            st.subheader("RSI(14)")
            st.line_chart(df[["RSI14"]])

            st.subheader("MACD Histogram")
            st.bar_chart(df[["MACD_hist"]])

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
else:
    st.info("👈 Soldan bir sembol seçip **Analiz Et** butonuna bas.")
