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

try:
    import anthropic
except ImportError:
    anthropic = None


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


def atr(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window).mean()


def support_resistance(df, lookbacks=(20, 50)):
    """Kısa ve orta vadeli en yakın destek/direnç seviyelerini döndürür."""
    levels = {}
    for lb in lookbacks:
        if len(df) >= lb:
            levels[lb] = {
                "resistance": float(df["High"].tail(lb).max()),
                "support": float(df["Low"].tail(lb).min()),
            }
    return levels


def fibonacci_levels(df, lookback=100):
    """Son `lookback` bar içindeki en yüksek/en düşük noktalar arasında
    standart Fibonacci düzeltme seviyelerini hesaplar."""
    window = df.tail(min(lookback, len(df)))
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    diff = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {r: swing_high - diff * r for r in ratios}
    return levels, swing_high, swing_low


def volatility_regime(df, window=14, lookback=100):
    """Güncel ATR'yi kendi geçmiş dağılımına göre değerlendirir."""
    atr_series = atr(df, window)
    recent = atr_series.dropna().tail(lookback)
    if recent.empty:
        return None
    current = recent.iloc[-1]
    percentile = (recent < current).mean() * 100
    if percentile >= 75:
        label = "Yüksek Volatilite"
    elif percentile <= 25:
        label = "Düşük Volatilite"
    else:
        label = "Normal Volatilite"
    return {"atr": float(current), "percentile": float(percentile), "label": label}


def volume_trend(df, short=20, long=60):
    """Kısa vadeli ortalama hacmi, uzun vadeli ortalama hacimle kıyaslar."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return None
    vol = df["Volume"]
    if len(vol) < long:
        return None
    short_avg = vol.tail(short).mean()
    long_avg = vol.tail(long).mean()
    if long_avg == 0 or np.isnan(long_avg):
        return None
    change_pct = (short_avg / long_avg - 1) * 100
    label = "Artan Katılım" if change_pct > 10 else ("Azalan Katılım" if change_pct < -10 else "Normal Katılım")
    return {"short_avg": float(short_avg), "long_avg": float(long_avg),
            "change_pct": float(change_pct), "label": label}


@st.cache_data(ttl=300)
def fetch_weekly_confirmation(symbol):
    """Haftalık zaman diliminde de trend aynı yönde mi kontrol eder."""
    try:
        wdf = yf.download(symbol, period="3y", interval="1wk", progress=False)
        if wdf.empty:
            return None
        if isinstance(wdf.columns, pd.MultiIndex):
            wdf.columns = [c[0] for c in wdf.columns]
        wdf = wdf.rename(columns=str.title).dropna()
        if len(wdf) < 20:
            return None
        _, wresult = analyze_trend(wdf, symbol=symbol)
        return wresult
    except Exception:
        return None


MACRO_TICKERS = {
    "Dolar Endeksi (DXY)": "DX=F",
    "ABD 10Y Tahvil Faizi": "^TNX",
    "USD/TRY": "TRY=X",
    "S&P 500": "^GSPC",
}


@st.cache_data(ttl=600)
def fetch_macro_snapshot():
    """Makro bağlam göstergelerinin son değerini ve son 1 aylık değişimini çeker."""
    rows = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.title).dropna()
            last_price = float(df["Close"].iloc[-1])
            month_ago_price = float(df["Close"].iloc[max(0, len(df) - 22)])
            change_pct = (last_price / month_ago_price - 1) * 100
            rows.append({
                "Gösterge": label,
                "Son Değer": round(last_price, 2),
                "1 Aylık Değişim": f"{change_pct:+.2f}%",
                "_change_raw": change_pct,
            })
        except Exception:
            continue
    return rows


# ============================================================
# BACKTEST MOTORU
# ============================================================
# Botun tekli analizde kullandığı AYNI ağırlıklandırılmış puanlama
# mantığını, her gün için (o günkü veriye kadar bakarak, ileriye
# sızıntı olmadan) hesaplayıp geçmişte üretmiş olacağı sinyalleri
# simüle eder.

def vectorized_trend_signal(df):
    """Her satır için (o ana kadarki veriyle) trend sinyalini ve güven
    skorunu hesaplar. Tüm göstergeler geriye dönük (rolling/ewm) olduğu
    için ileriye veri sızıntısı yoktur."""
    df = df.copy()
    close = df["Close"]
    df["SMA50"] = sma(close, 50)
    df["SMA200"] = sma(close, 200)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(close)
    df["ROC10"] = roc(close, 10)
    df["LR_Slope_%"], df["LR_R2"] = linear_regression_channel(close, window=50)

    cond_strong_up = (df["Close"] > df["SMA50"]) & (df["SMA50"] > df["SMA200"])
    cond_strong_down = (df["Close"] < df["SMA50"]) & (df["SMA50"] < df["SMA200"])
    cond_weak_up = df["Close"] > df["SMA50"]
    ma_score = np.where(
        cond_strong_up, 1.0,
        np.where(cond_strong_down, -1.0, np.where(cond_weak_up, 0.4, -0.4))
    )

    macd_bull = df["MACD"] > df["MACD_signal"]
    roc_bull = df["ROC10"] > 0
    mom_score = np.where(
        macd_bull & roc_bull, 1.0,
        np.where(~macd_bull & ~roc_bull, -1.0, np.where(macd_bull, 0.3, -0.3))
    )

    slope = df["LR_Slope_%"]
    r2 = df["LR_R2"].fillna(0)
    channel_score = np.where(
        slope > 0.05, np.minimum(1.0, r2 + 0.3),
        np.where(slope < -0.05, -np.minimum(1.0, r2 + 0.3), 0)
    )

    score = ma_score * 35 + mom_score * 30 + channel_score * 35
    confidence = np.minimum(100, np.abs(score) + r2 * 15)

    df["Signal"] = np.where(score > 15, 1, np.where(score < -15, -1, 0))
    df["Confidence"] = confidence
    return df


def run_backtest(raw_df, min_confidence=0, allow_short=False):
    """Botun sinyaline dayalı basit bir long/short simülasyonu çalıştırır.
    Sinyal t günü kapanışına kadarki veriyle hesaplanır, pozisyon t+1
    gününün getirisini yakalar (ileriye sızıntı yok)."""
    df = vectorized_trend_signal(raw_df)
    df = df.dropna(subset=["SMA200"]).copy()
    if len(df) < 30:
        return None

    signal = df["Signal"].to_numpy().astype(float)
    if min_confidence > 0:
        signal = np.where(df["Confidence"].to_numpy() >= min_confidence, signal, 0)

    if not allow_short:
        signal = np.where(signal == 1, 1, 0)

    position = pd.Series(signal, index=df.index)
    daily_return = df["Close"].pct_change().fillna(0)
    strat_return = position.shift(1).fillna(0) * daily_return

    equity = (1 + strat_return).cumprod()
    buyhold_equity = (1 + daily_return).cumprod()

    # --- İşlem listesi çıkarma ---
    trades = []
    current_pos = 0
    entry_price = None
    entry_date = None
    for i in range(len(df)):
        pos = position.iloc[i]
        price = float(df["Close"].iloc[i])
        d = df.index[i]
        if pos != current_pos:
            if current_pos != 0:
                ret = (price / entry_price - 1) * current_pos * 100
                trades.append({
                    "Giriş": entry_date.strftime("%Y-%m-%d"),
                    "Çıkış": d.strftime("%Y-%m-%d"),
                    "Yön": "LONG" if current_pos == 1 else "SHORT",
                    "Getiri %": round(ret, 2),
                })
            if pos != 0:
                entry_price = price
                entry_date = d
            current_pos = pos
    if current_pos != 0:
        price = float(df["Close"].iloc[-1])
        ret = (price / entry_price - 1) * current_pos * 100
        trades.append({
            "Giriş": entry_date.strftime("%Y-%m-%d"),
            "Çıkış": df.index[-1].strftime("%Y-%m-%d") + " (açık)",
            "Yön": "LONG" if current_pos == 1 else "SHORT",
            "Getiri %": round(ret, 2),
        })

    trades_df = pd.DataFrame(trades)

    total_return = (equity.iloc[-1] - 1) * 100
    buyhold_return = (buyhold_equity.iloc[-1] - 1) * 100

    if len(trades_df) > 0:
        wins = trades_df[trades_df["Getiri %"] > 0]
        losses = trades_df[trades_df["Getiri %"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins["Getiri %"].mean() if len(wins) else 0.0
        avg_loss = losses["Getiri %"].mean() if len(losses) else 0.0
        loss_sum = abs(losses["Getiri %"].sum())
        profit_factor = (wins["Getiri %"].sum() / loss_sum) if loss_sum > 0 else np.nan
    else:
        win_rate = avg_win = avg_loss = 0.0
        profit_factor = np.nan

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "equity": equity,
        "buyhold_equity": buyhold_equity,
        "drawdown": drawdown,
        "trades": trades_df,
        "stats": {
            "total_return": total_return,
            "buyhold_return": buyhold_return,
            "num_trades": len(trades_df),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        },
    }


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
        "BIST 30 Endeksi": "XU030.IS",
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

    # --------------------------------------------------------------
    # DERİNLEMESİNE TEKNİK ANALİZ
    # --------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Derinlemesine Teknik Analiz")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**Destek / Direnç Seviyeleri**")
        sr = support_resistance(df, lookbacks=(20, 50))
        for lb, levels in sr.items():
            etiket = "Kısa Vadeli (20 bar)" if lb == 20 else "Orta Vadeli (50 bar)"
            st.markdown(
                f"- {etiket}: Direnç **{levels['resistance']:.4f}** / "
                f"Destek **{levels['support']:.4f}**"
            )

        st.markdown("**Volatilite Durumu (ATR)**")
        vol_regime = volatility_regime(df)
        if vol_regime:
            st.markdown(
                f"- ATR(14): **{vol_regime['atr']:.4f}** — "
                f"{vol_regime['label']} (yüzdelik dilim: {vol_regime['percentile']:.0f})"
            )
        else:
            st.caption("Volatilite hesaplamak için yeterli veri yok.")

    with dcol2:
        st.markdown("**Fibonacci Düzeltme Seviyeleri** (son 100 bar)")
        fib_levels, swing_high, swing_low = fibonacci_levels(df, lookback=100)
        st.caption(f"Zirve: {swing_high:.4f} — Dip: {swing_low:.4f}")
        for ratio, price in fib_levels.items():
            st.markdown(f"- %{ratio*100:.1f}: **{price:.4f}**")

        st.markdown("**Hacim Analizi**")
        vtrend = volume_trend(df)
        if vtrend:
            st.markdown(
                f"- Kısa vadeli ort. hacim, uzun vadeliye göre "
                f"**{vtrend['change_pct']:+.1f}%** — {vtrend['label']}"
            )
        else:
            st.caption("Bu sembol için hacim verisi yok veya yetersiz.")

    st.markdown("**Çoklu Zaman Dilimi Onayı (Haftalık)**")
    weekly = fetch_weekly_confirmation(symbol)
    if weekly:
        same_direction = weekly["trend"] == result["trend"]
        icon = "✅" if same_direction else "⚠️"
        st.markdown(
            f"{icon} Haftalık grafikte trend: **{weekly['trend']}** "
            f"(güven: {weekly['confidence']:.0f}/100) — "
            + ("günlük trendle aynı yönde, teyit ediyor." if same_direction
               else "günlük trendle **çelişiyor**, dikkatli yorumla.")
        )
    else:
        st.caption("Haftalık veri çekilemedi veya yetersiz.")

    st.caption(
        "Not: Bu bölüm ek teknik göstergeler sunar, yorum yapmaz — "
        "seviyelerin ne anlama geldiğine sen karar verirsin."
    )


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

with st.expander("🌍 Makro Bağlam (Dolar Endeksi, Faiz, USD/TRY, S&P 500)", expanded=False):
    if yf is None:
        st.caption("yfinance kurulu değil.")
    else:
        macro_rows = fetch_macro_snapshot()
        if macro_rows:
            macro_df = pd.DataFrame(macro_rows).drop(columns=["_change_raw"])
            st.dataframe(macro_df, use_container_width=True, hide_index=True)
            st.caption(
                "Bu veriler sadece bağlam sağlar, yorum içermez. Genel kural olarak: "
                "Dolar Endeksi ve ABD tahvil faizleri yükseldiğinde altın genelde baskı "
                "altında kalır; USD/TRY yükseldiğinde BIST endeksleri döviz bazlı yatırımcı "
                "için farklı bir tablo çizebilir. Yorumlama sana aittir."
            )
        else:
            st.caption("Makro veriler şu an çekilemedi.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Tekli Analiz", "📊 Çoklu Varlık Tarama", "🤖 AI Piyasa Koçu", "🧪 Backtest"]
)

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

# --------------------------------------------------------------
# SEKME 3: AI PİYASA KOÇU (gerçek AI sohbet, Anthropic API ile)
# --------------------------------------------------------------
with tab3:
    st.subheader("AI Piyasa Koçuna Sor")
    st.caption(
        "Piyasalarla ilgili serbest metinle soru sorabilirsin — "
        "örn. \"BIST 30 nasıl hareket ediyor?\", \"Altın için bu hafta önemli ne var?\""
    )

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None

    if anthropic is None:
        st.error(
            "`anthropic` kütüphanesi kurulu değil. `requirements.txt` dosyasına "
            "`anthropic` satırını eklediğinden emin ol."
        )
    elif not api_key:
        st.info(
            "🔑 AI Piyasa Koçu'nu kullanmak için bir Anthropic API anahtarı gerekiyor.\n\n"
            "**Nasıl eklenir:**\n"
            "1. [platform.claude.com](https://platform.claude.com) üzerinden ücretsiz bir hesap aç, "
            "ödeme yöntemi ekleyip en az 5 dolarlık kredi yükle\n"
            "2. Settings → API Keys bölümünden yeni bir anahtar oluştur\n"
            "3. Streamlit Cloud'da uygulamanı yönet (Manage app) → sağ altta **⋮ (üç nokta)** → "
            "**Settings → Secrets** bölümüne şunu ekle:\n\n"
            "```\nANTHROPIC_API_KEY = \"buraya-anahtarini-yapistir\"\n```\n\n"
            "Kaydettikten sonra bu sekme otomatik aktif olur."
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Sorunu yaz...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Düşünüyor..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=1000,
                            system=(
                                "Sen Türkçe konuşan bir piyasa analiz koçusun. Kullanıcının "
                                "trend following stratejisiyle ilgilenen bir bireysel trader "
                                "olduğunu biliyorsun. Sorulara net, kısa ve anlaşılır cevaplar "
                                "ver. Güncel veri gerekiyorsa web araması yap. Asla kesin 'al' "
                                "veya 'sat' komutu verme — bunun yerine gözlem ve analiz paylaş. "
                                "Her cevabının sonunda kısaca bunun yatırım tavsiyesi olmadığını "
                                "hatırlat."
                            ),
                            tools=[{"type": "web_search_20250305", "name": "web_search"}],
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history
                            ],
                        )
                        answer = "\n".join(
                            block.text for block in response.content if block.type == "text"
                        ).strip()
                        if not answer:
                            answer = "Bir cevap üretemedim, lütfen soruyu farklı şekilde tekrar dener misin?"
                    except Exception as e:
                        answer = f"Hata oluştu: {e}"

                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("🗑️ Sohbeti Temizle"):
                st.session_state.chat_history = []
                st.rerun()

# --------------------------------------------------------------
# SEKME 4: BACKTEST
# --------------------------------------------------------------
with tab4:
    st.subheader("Botun Sinyalini Geçmiş Veriyle Test Et")
    st.caption(
        "Bot, aynı analiz mantığını (MA + Momentum + Kanal) geçmişte her gün "
        "yeniden hesaplayarak o tarihte ne diyeceğini simüle eder. İleriye dönük "
        "veri sızıntısı yoktur — her gün sadece o güne kadarki veriyle karar verir."
    )

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        bt_symbol = st.text_input("Sembol", value="GC=F", key="bt_symbol",
                                   help="Örn: GC=F (Altın), XU030.IS (BIST 30)")
    with bcol2:
        bt_period = st.selectbox("Test Dönemi", ["1y", "2y", "5y", "max"], index=1, key="bt_period")
    with bcol3:
        bt_min_conf = st.slider("Minimum Güven Skoru Filtresi", 0, 100, 0, key="bt_min_conf",
                                 help="Bu skorun altındaki sinyaller işlem açmaz (0 = filtre yok)")

    bt_allow_short = st.checkbox(
        "Düşüş sinyallerinde de işlem aç (Long + Short)",
        value=False,
        help="Kapalıysa bot sadece yükseliş trendinde pozisyon açar (Long-only), "
             "düşüş sinyallerinde nakitte bekler.",
    )

    bt_run = st.button("🧪 Backtest Çalıştır", type="primary")

    if bt_run:
        if yf is None:
            st.error("yfinance kurulu değil.")
        else:
            try:
                with st.spinner(f"{bt_symbol} için {bt_period} geçmiş veri test ediliyor..."):
                    bt_raw = fetch_data(bt_symbol, period=bt_period, interval="1d")
                    bt_result = run_backtest(bt_raw, min_confidence=bt_min_conf, allow_short=bt_allow_short)

                if bt_result is None:
                    st.warning("Yeterli veri yok (en az ~200 günlük geçmiş gerekiyor, SMA200 için).")
                else:
                    stats = bt_result["stats"]

                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Strateji Getirisi", f"{stats['total_return']:.1f}%")
                    m2.metric("Al-ve-Tut (Buy&Hold)", f"{stats['buyhold_return']:.1f}%",
                               delta=f"{stats['total_return'] - stats['buyhold_return']:.1f}% fark")
                    m3.metric("İşlem Sayısı", f"{stats['num_trades']}")
                    m4.metric("Maksimum Düşüş", f"{stats['max_drawdown']:.1f}%")

                    m5, m6, m7 = st.columns(3)
                    m5.metric("Kazanma Oranı", f"{stats['win_rate']:.1f}%")
                    m6.metric("Ort. Kazanç / Ort. Kayıp",
                               f"{stats['avg_win']:.1f}% / {stats['avg_loss']:.1f}%")
                    pf = stats["profit_factor"]
                    m7.metric("Kâr Faktörü", "∞" if (pf != pf) else f"{pf:.2f}")

                    st.divider()
                    st.subheader("Getiri Eğrisi: Strateji vs Al-ve-Tut")
                    equity_compare = pd.DataFrame({
                        "Strateji": bt_result["equity"],
                        "Al-ve-Tut": bt_result["buyhold_equity"],
                    })
                    st.line_chart(equity_compare)

                    st.subheader("Düşüş (Drawdown) Grafiği")
                    st.area_chart(bt_result["drawdown"])

                    st.subheader(f"İşlem Listesi ({len(bt_result['trades'])} işlem)")
                    if len(bt_result["trades"]) > 0:
                        st.dataframe(bt_result["trades"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Bu dönemde hiç işlem üretilmedi.")

                    st.info(
                        "📌 **Nasıl okunmalı:** 'Strateji Getirisi' > 'Al-ve-Tut' ise, bot sinyali "
                        "bu dönemde basitçe alıp beklemekten daha iyi performans göstermiş demektir. "
                        "Az işlem sayısı ve büyük maksimum düşüş, sonucun tesadüfe daha açık olabileceğine "
                        "işaret eder. Bu geçmiş performans, gelecekte aynı sonucu garanti etmez.",
                        icon="📌",
                    )

            except Exception as e:
                st.error(f"Hata oluştu: {e}")
    else:
        st.info("👆 Sembol ve ayarları seçip **Backtest Çalıştır** butonuna bas.")
