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
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    yf = None


# ============================================================
# EKONOMİK TAKVİM (FED Faiz Kararları + ABD İstihdam Verisi)
# ============================================================
# Not: FOMC tarihleri Fed'in resmi takviminden alınmıştır (2026).
# NFP (Tarım Dışı İstihdam) her ayın ilk Cuma günü açıklanır (kural sabit).
# Bu bir haber akışı değil, bilinen takvim tarihlerinin gösterimidir.

FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # Cuma = 4
    return d + timedelta(days=offset)


def get_upcoming_events(n_months: int = 3):
    """Bugünden itibaren her ay için en fazla 2 önemli olay döndürür:
    o ay bir FOMC toplantısı varsa onu, ayrıca ayın NFP (istihdam) tarihini."""
    today = date.today()
    events = []
    year, month = today.year, today.month

    for _ in range(n_months):
        month_events = []

        fomc_this_month = [d for d in FOMC_DATES_2026 if d.year == year and d.month == month]
        for d in fomc_this_month:
            month_events.append(("🏦 FED Faiz Kararı (FOMC)", d))

        nfp_date = first_friday(year, month)
        month_events.append(("👷 ABD Tarım Dışı İstihdam (NFP)", nfp_date))

        # Sadece bugün ve sonrası, ay başına en fazla 2 olay
        month_events = [e for e in month_events if e[1] >= today][:2]
        events.extend(month_events)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return sorted(events, key=lambda x: x[1])


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
# HAZIR VARLIK LİSTELERİ (Çoklu Tarama için)
# ============================================================

ASSET_LISTS = {
    "Emtialar": {
        "Altın": "GC=F",
        "Gümüş": "SI=F",
        "Petrol (WTI)": "CL=F",
        "Doğalgaz": "NG=F",
        "Bakır": "HG=F",
    },
    "Kripto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "BNB": "BNB-USD",
        "XRP": "XRP-USD",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "USD/TRY": "TRY=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
    },
    "BIST": {
        "Türk Hava Yolları": "THYAO.IS",
        "Aselsan": "ASELS.IS",
        "Garanti BBVA": "GARAN.IS",
        "Koç Holding": "KCHOL.IS",
    },
}


def render_single_result(df, result, symbol):
    trend_color = {
        "YÜKSELİŞ TRENDİ": "green",
        "DÜŞÜŞ TRENDİ": "red",
        "YATAY / BELİRSİZ": "orange",
    }[result["trend"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Son Fiyat", f"{result['last_price']:.4f}")
    col2.markdown(f"### Trend: :{trend_color}[{result['trend']}]")
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


# ============================================================
# ARAYÜZ (STREAMLIT)
# ============================================================

st.set_page_config(page_title="Trend Following Analiz Botu", page_icon="📈", layout="wide")

st.title("📈 Trend Following Analiz Botu")
st.caption("Hareketli ortalama + momentum + kanal analiziyle trend yönü tespiti")
st.markdown("**Geliştirici:** Yusuf İslam Akbulut")

st.warning(
    "⚠️ Bu araç yalnızca teknik analiz amaçlıdır, yatırım tavsiyesi değildir. "
    "Yatırım kararlarının sorumluluğu tamamen kullanıcıya aittir.",
    icon="⚠️",
)

with st.expander("📅 Yaklaşan Önemli Ekonomik Olaylar", expanded=False):
    upcoming = get_upcoming_events(n_months=3)
    if upcoming:
        for label, d in upcoming:
            days_left = (d - date.today()).days
            gun_text = "Bugün" if days_left == 0 else f"{days_left} gün sonra"
            st.markdown(f"- **{label}** — {d.strftime('%d %B %Y')} ({gun_text})")
        st.caption(
            "Bu tarihlerde piyasa volatilitesi artabilir, sinyal güvenilirliği "
            "geçici olarak düşebilir. Sabit takvim kuralına dayanır (canlı haber akışı değildir)."
        )
    else:
        st.caption("Önümüzdeki dönem için gösterilecek olay bulunamadı.")

tab1, tab2 = st.tabs(["🔍 Tekli Analiz", "📊 Çoklu Varlık Tarama"])

# --------------------------------------------------------------
# SEKME 1: TEKLİ ANALİZ (eski davranış)
# --------------------------------------------------------------
with tab1:
    with st.sidebar:
        st.header("Tekli Analiz Ayarları")
        symbol = st.text_input("Sembol", value="BTC-USD",
                                help="Örn: BTC-USD, THYAO.IS, AAPL, EURUSD=X, GC=F")
        period = st.selectbox("Zaman Aralığı", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="single_period")
        interval = st.selectbox("Mum Periyodu", ["1d", "1h", "1wk"], index=0, key="single_interval")
        run = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

    if run:
        if yf is None:
            st.error("yfinance kurulu değil.")
        else:
            try:
                with st.spinner(f"{symbol} için veri çekiliyor ve analiz ediliyor..."):
                    raw_df = fetch_data(symbol, period=period, interval=interval)
                    df, result = analyze_trend(raw_df, symbol=symbol)
                render_single_result(df, result, symbol)
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
    else:
        st.info("👈 Soldan bir sembol seçip **Analiz Et** butonuna bas.")

# --------------------------------------------------------------
# SEKME 2: ÇOKLU VARLIK TARAMA
# --------------------------------------------------------------
with tab2:
    st.subheader("Birden fazla varlığı aynı anda tara")

    scan_source = st.radio(
        "Taranacak liste",
        ["Hazır liste seç", "Kendi listemi gireyim"],
        horizontal=True,
    )

    symbols_to_scan = {}

    if scan_source == "Hazır liste seç":
        list_name = st.selectbox("Kategori", list(ASSET_LISTS.keys()))
        symbols_to_scan = ASSET_LISTS[list_name]
        st.caption("Taranacaklar: " + ", ".join(f"{k} ({v})" for k, v in symbols_to_scan.items()))
    else:
        custom_input = st.text_area(
            "Sembolleri virgülle ayırarak gir",
            value="GC=F, SI=F, BTC-USD",
            help="Örn: GC=F, SI=F, BTC-USD, THYAO.IS",
        )
        raw_symbols = [s.strip() for s in custom_input.split(",") if s.strip()]
        symbols_to_scan = {s: s for s in raw_symbols}

    col_a, col_b = st.columns(2)
    with col_a:
        scan_period = st.selectbox("Zaman Aralığı", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="scan_period")
    with col_b:
        scan_interval = st.selectbox("Mum Periyodu", ["1d", "1h", "1wk"], index=0, key="scan_interval")

    scan_run = st.button("📊 Taramayı Başlat", type="primary")

    if scan_run:
        if yf is None:
            st.error("yfinance kurulu değil.")
        elif not symbols_to_scan:
            st.warning("Taranacak sembol bulunamadı.")
        else:
            rows = []
            errors = []
            progress = st.progress(0, text="Taranıyor...")
            total = len(symbols_to_scan)

            for i, (label, sym) in enumerate(symbols_to_scan.items()):
                try:
                    raw_df = fetch_data(sym, period=scan_period, interval=scan_interval)
                    _, result = analyze_trend(raw_df, symbol=sym)
                    rows.append({
                        "Varlık": label,
                        "Sembol": sym,
                        "Trend": result["trend"],
                        "Güven Skoru": round(result["confidence"]),
                        "Son Fiyat": round(result["last_price"], 4),
                        "RSI(14)": round(result["rsi_value"], 1),
                        "Kanal": result["channel_direction"],
                    })
                except Exception as e:
                    errors.append(f"{label} ({sym}): {e}")
                progress.progress((i + 1) / total, text=f"Taranıyor... {i + 1}/{total}")

            progress.empty()

            if rows:
                result_df = pd.DataFrame(rows).sort_values("Güven Skoru", ascending=False)

                def highlight_trend(val):
                    if val == "YÜKSELİŞ TRENDİ":
                        return "color: green; font-weight: bold"
                    elif val == "DÜŞÜŞ TRENDİ":
                        return "color: red; font-weight: bold"
                    return "color: orange"

                st.dataframe(
                    result_df.style.map(highlight_trend, subset=["Trend"]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Tablo güven skoruna göre büyükten küçüğe sıralanmıştır.")
            else:
                st.warning("Hiçbir sembol için sonuç alınamadı.")

            if errors:
                with st.expander(f"⚠️ {len(errors)} sembolde hata oluştu (detay için tıkla)"):
                    for err in errors:
                        st.write(f"- {err}")
