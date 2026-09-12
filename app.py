"""
ALTIN (GOLD) TREND FOLLOWING ANALİZ BOTU — WEB UYGULAMASI
============================================================
Sadece altın ve altın ailesi enstrümanlarına (Ons Altın, Altın Vadeli
İşlem, Gümüş, Altın Madenciliği ETF'leri) odaklanan bir analiz botu.

Yerelde çalıştırmak için:
    pip install streamlit pandas numpy yfinance
    streamlit run app.py
"""

import warnings
from datetime import date, datetime, timedelta

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
# Bu iki veri, altın fiyatını en çok etkileyen ABD makro olaylarıdır:
# faiz kararları doların ve reel faizlerin yönünü, istihdam verisi de
# FED'in faiz politikası beklentisini şekillendirir.

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
    """Bugünden itibaren her ay için en fazla 2 önemli olay döndürür."""
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
        month_events = [e for e in month_events if e[1] >= today][:2]
        events.extend(month_events)
        month += 1
        if month > 12:
            month = 1
            year += 1

    return sorted(events, key=lambda x: x[1])


# ============================================================
# ALTIN HABERLERİ (Yahoo Finance haber akışı — ücretsiz)
# ============================================================

@st.cache_data(ttl=900)
def fetch_gold_news(max_items=8):
    """Altın vadeli işlem sembolüne bağlı güncel haber başlıklarını çeker.
    yfinance'in haber şeması sürüme göre değişebildiği için iki farklı
    formatı da (eski/yeni) esnek şekilde okumaya çalışır."""
    if yf is None:
        return []
    try:
        raw_news = yf.Ticker("GC=F").news or []
    except Exception:
        return []

    items = []
    for n in raw_news[:max_items]:
        title = n.get("title")
        publisher = n.get("publisher")
        link = n.get("link")
        ts = n.get("providerPublishTime")

        if not title and "content" in n:
            content = n.get("content", {})
            title = content.get("title")
            provider = content.get("provider") or {}
            publisher = provider.get("displayName")
            url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            link = url_obj.get("url")
            pub_date_str = content.get("pubDate")
            if pub_date_str:
                try:
                    dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except Exception:
                    ts = None

        if not title:
            continue

        time_str = None
        if ts:
            try:
                time_str = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
            except Exception:
                time_str = None

        items.append({
            "title": title,
            "publisher": publisher or "Bilinmeyen kaynak",
            "link": link,
            "time": time_str,
        })
    return items


# ============================================================
# İNDİKATÖR FONKSİYONLARI
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
    levels = {}
    for lb in lookbacks:
        if len(df) >= lb:
            levels[lb] = {
                "resistance": float(df["High"].tail(lb).max()),
                "support": float(df["Low"].tail(lb).min()),
            }
    return levels


def fibonacci_levels(df, lookback=100):
    window = df.tail(min(lookback, len(df)))
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    diff = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {r: swing_high - diff * r for r in ratios}
    return levels, swing_high, swing_low


def volatility_regime(df, window=14, lookback=100):
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


# ============================================================
# MAKRO BAĞLAM (sadece altını etkileyen faktörler)
# ============================================================

MACRO_TICKERS = {
    "Altın (Ons - GC=F)": "GC=F",
    "Gümüş (Korele Emtia)": "SI=F",
    "Dolar Endeksi (DXY)": "DX=F",
    "ABD 10Y Tahvil Faizi": "^TNX",
    "VIX (Korku Endeksi)": "^VIX",
}


@st.cache_data(ttl=600)
def fetch_macro_snapshot():
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
            })
        except Exception:
            continue
    return rows


# ============================================================
# BACKTEST MOTORU
# ============================================================

def vectorized_trend_signal(df):
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


def run_backtest(raw_df, min_confidence=0, allow_short=False, cash_rate_annual=0.0):
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

    daily_cash_return = (cash_rate_annual / 100) / 365
    position_shifted = position.shift(1).fillna(0).to_numpy()
    strat_return_arr = np.where(
        position_shifted == 0,
        daily_cash_return,
        position_shifted * daily_return.to_numpy(),
    )
    strat_return = pd.Series(strat_return_arr, index=df.index)
    equity = (1 + strat_return).cumprod()

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
        "drawdown": drawdown,
        "trades": trades_df,
        "stats": {
            "total_return": total_return,
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
# ALTIN AİLESİ ENSTRÜMANLARI (Çoklu Tarama için)
# ============================================================

GOLD_FAMILY = {
    "Altın Vadeli İşlem (COMEX)": "GC=F",
    "Ons Altın (Spot)": "XAUUSD=X",
    "Gümüş Vadeli İşlem": "SI=F",
    "Altın ETF (SPDR Gold Shares)": "GLD",
    "Altın Madenciliği ETF (GDX)": "GDX",
    "Küçük Altın Madencileri ETF (GDXJ)": "GDXJ",
    "Newmont Corporation": "NEM",
    "Barrick Gold": "GOLD",
}


def render_single_result(df, result, symbol):
    trend_emoji = {
        "YÜKSELİŞ TRENDİ": "🟢",
        "DÜŞÜŞ TRENDİ": "🔴",
        "YATAY / BELİRSİZ": "🟡",
    }[result["trend"]]
    trend_color = {
        "YÜKSELİŞ TRENDİ": "green",
        "DÜŞÜŞ TRENDİ": "red",
        "YATAY / BELİRSİZ": "orange",
    }[result["trend"]]

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Son Fiyat", f"{result['last_price']:.4f}")
        col2.markdown(f"### {trend_emoji} :{trend_color}[{result['trend']}]")
        col3.metric("Güven Skoru", f"{result['confidence']:.0f} / 100")

    st.write("")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**📊 Kanal Yönü**\n\n{result['channel_direction']}  \n(R²={result['channel_r2']:.2f})")
        c2.markdown(f"**📈 MA Sinyali**\n\n{result['ma_signal']}")
        c3.markdown(f"**⚡ Momentum**\n\n{result['momentum_signal']}  \nRSI(14): {result['rsi_value']:.1f}")

    st.divider()
    st.subheader("Fiyat Grafiği")
    chart_df = df[["Close", "SMA50", "SMA200", "Donchian_Upper", "Donchian_Lower"]]
    st.line_chart(chart_df)

    st.subheader("RSI(14)")
    st.line_chart(df[["RSI14"]])

    st.subheader("MACD Histogram")
    st.bar_chart(df[["MACD_hist"]])

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

st.set_page_config(page_title="Altın Trend Following Botu", page_icon="🥇", layout="wide")

st.title("🥇 Altın (Gold) Trend Following Analiz Botu")
st.caption(
    "Sadece altın ve altın ailesi enstrümanlarına odaklı: teknik analiz · backtest · "
    "pozisyon hesaplayıcı · güncel haberler  —  **Geliştirici:** Yusuf İslam Akbulut"
)

st.warning(
    "⚠️ Bu araç yalnızca teknik analiz amaçlıdır, yatırım tavsiyesi değildir. "
    "Yatırım kararlarının sorumluluğu tamamen kullanıcıya aittir.",
    icon="⚠️",
)

with st.expander("📰 Altına Dair Güncel Haberler", expanded=True):
    if yf is None:
        st.caption("yfinance kurulu değil.")
    else:
        news_items = fetch_gold_news(max_items=8)
        if news_items:
            for item in news_items:
                time_part = f" · {item['time']}" if item["time"] else ""
                if item["link"]:
                    st.markdown(f"- [{item['title']}]({item['link']}) — *{item['publisher']}*{time_part}")
                else:
                    st.markdown(f"- {item['title']} — *{item['publisher']}*{time_part}")
        else:
            st.caption("Şu an haber çekilemedi. Birkaç dakika sonra tekrar dene.")

with st.expander("📅 Yaklaşan Önemli Ekonomik Olaylar", expanded=False):
    upcoming = get_upcoming_events(n_months=3)
    if upcoming:
        for label, d in upcoming:
            days_left = (d - date.today()).days
            gun_text = "Bugün" if days_left == 0 else f"{days_left} gün sonra"
            st.markdown(f"- **{label}** — {d.strftime('%d %B %Y')} ({gun_text})")
        st.caption(
            "Bu tarihlerde altın fiyatında volatilite artabilir — faiz kararları doları "
            "ve reel faizleri, istihdam verisi de FED beklentisini doğrudan etkiler."
        )
    else:
        st.caption("Önümüzdeki dönem için gösterilecek olay bulunamadı.")

with st.expander("🌍 Makro Bağlam (Dolar Endeksi, Faiz, VIX, Gümüş)", expanded=False):
    if yf is None:
        st.caption("yfinance kurulu değil.")
    else:
        macro_rows = fetch_macro_snapshot()
        if macro_rows:
            macro_df = pd.DataFrame(macro_rows)
            st.dataframe(macro_df, use_container_width=True, hide_index=True)
            st.caption(
                f"🕒 Veriler canlı piyasa verisidir (en fazla 10 dakika önbelleklidir). "
                f"Son kontrol: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            )
            st.caption(
                "Genel kural: Dolar Endeksi ve ABD tahvil faizleri yükseldiğinde altın "
                "genelde baskı altında kalır; VIX (korku endeksi) yükseldiğinde altın "
                "güvenli liman talebiyle destek bulabilir. Yorumlama sana aittir."
            )
        else:
            st.caption("Makro veriler şu an çekilemedi.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Tekli Analiz", "📊 Altın Ailesi Tarama", "🧪 Backtest", "🧮 Pozisyon Hesaplayıcı"]
)

# --------------------------------------------------------------
# SEKME 1: TEKLİ ANALİZ
# --------------------------------------------------------------
with tab1:
    st.subheader("Altın Enstrümanını Analiz Et")

    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        symbol = st.selectbox(
            "Enstrüman", list(GOLD_FAMILY.values()),
            format_func=lambda x: next(k for k, v in GOLD_FAMILY.items() if v == x),
            index=0, key="single_symbol",
        )
    with ic2:
        period = st.selectbox("Zaman Aralığı", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="single_period")
    with ic3:
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
        st.info("👆 Bir enstrüman seçip **Analiz Et** butonuna bas.")

# --------------------------------------------------------------
# SEKME 2: ALTIN AİLESİ TARAMA
# --------------------------------------------------------------
with tab2:
    st.subheader("Altın Ailesindeki Tüm Enstrümanları Tara")
    st.caption(
        "Altın, gümüş ve altınla ilişkili hisse/ETF'leri aynı anda tarayıp hangisinde "
        "daha net bir trend olduğunu karşılaştırır."
    )
    st.caption("Taranacaklar: " + ", ".join(f"{k} ({v})" for k, v in GOLD_FAMILY.items()))

    col_a, col_b = st.columns(2)
    with col_a:
        scan_period = st.selectbox("Zaman Aralığı", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="scan_period")
    with col_b:
        scan_interval = st.selectbox("Mum Periyodu", ["1d", "1h", "1wk"], index=0, key="scan_interval")

    scan_run = st.button("📊 Taramayı Başlat", type="primary")

    if scan_run:
        if yf is None:
            st.error("yfinance kurulu değil.")
        else:
            rows = []
            errors = []
            progress = st.progress(0, text="Taranıyor...")
            total = len(GOLD_FAMILY)

            for i, (label, sym) in enumerate(GOLD_FAMILY.items()):
                try:
                    raw_df = fetch_data(sym, period=scan_period, interval=scan_interval)
                    _, result = analyze_trend(raw_df, symbol=sym)
                    rows.append({
                        "Enstrüman": label,
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
                st.warning("Hiçbir enstrüman için sonuç alınamadı.")

            if errors:
                with st.expander(f"⚠️ {len(errors)} enstrümanda hata oluştu (detay için tıkla)"):
                    for err in errors:
                        st.write(f"- {err}")
    else:
        st.info("👆 Ayarları seçip **Taramayı Başlat** butonuna bas.")

# --------------------------------------------------------------
# SEKME 3: BACKTEST
# --------------------------------------------------------------
with tab3:
    st.subheader("Botun Sinyalini Geçmiş Veriyle Test Et")
    st.caption(
        "Bot, aynı analiz mantığını (MA + Momentum + Kanal) geçmişte her gün "
        "yeniden hesaplayarak o tarihte ne diyeceğini simüle eder. İleriye dönük "
        "veri sızıntısı yoktur — her gün sadece o güne kadarki veriyle karar verir."
    )

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        bt_symbol = st.selectbox(
            "Enstrüman", list(GOLD_FAMILY.values()),
            format_func=lambda x: next(k for k, v in GOLD_FAMILY.items() if v == x),
            index=0, key="bt_symbol",
        )
    with bcol2:
        bt_period = st.selectbox("Test Dönemi", ["1y", "2y", "5y", "max"], index=1, key="bt_period")
    with bcol3:
        bt_min_conf = st.slider("Minimum Güven Skoru Filtresi", 0, 100, 0, key="bt_min_conf",
                                 help="Bu skorun altındaki sinyaller işlem açmaz (0 = filtre yok)")

    bcol4, bcol5 = st.columns(2)
    with bcol4:
        bt_allow_short = st.checkbox(
            "Düşüş sinyallerinde de işlem aç (Long + Short)",
            value=False,
            help="Kapalıysa bot sadece yükseliş trendinde pozisyon açar (Long-only), "
                 "düşüş sinyallerinde nakitte bekler.",
        )
    with bcol5:
        bt_cash_rate = st.number_input(
            "Nakitte Bekleme Faizi (Yıllık %)",
            min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            help="Pozisyon kapalıyken (nakitte) paranın bir para piyasası fonu/mevduatta "
                 "kazanacağı varsayılan yıllık faiz. 0 = nakit hiç getiri kazanmıyor varsayımı.",
        )

    bt_run = st.button("🧪 Backtest Çalıştır", type="primary")

    if bt_run:
        if yf is None:
            st.error("yfinance kurulu değil.")
        else:
            try:
                with st.spinner(f"{bt_symbol} için {bt_period} geçmiş veri test ediliyor..."):
                    bt_raw = fetch_data(bt_symbol, period=bt_period, interval="1d")
                    bt_result = run_backtest(
                        bt_raw, min_confidence=bt_min_conf,
                        allow_short=bt_allow_short, cash_rate_annual=bt_cash_rate,
                    )

                if bt_result is None:
                    st.warning("Yeterli veri yok (en az ~200 günlük geçmiş gerekiyor, SMA200 için).")
                else:
                    stats = bt_result["stats"]

                    st.divider()
                    with st.container(border=True):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Toplam Getiri", f"{stats['total_return']:.1f}%")
                        m2.metric("İşlem Sayısı", f"{stats['num_trades']}")
                        m3.metric("Kazanma Oranı", f"{stats['win_rate']:.1f}%")
                        m4.metric("Maksimum Düşüş", f"{stats['max_drawdown']:.1f}%")

                        m5, m6 = st.columns(2)
                        m5.metric("Ort. Kazanç / Ort. Kayıp",
                                   f"{stats['avg_win']:.1f}% / {stats['avg_loss']:.1f}%")
                        pf = stats["profit_factor"]
                        m6.metric("Kâr Faktörü", "∞" if (pf != pf) else f"{pf:.2f}")

                    st.divider()
                    st.subheader("Getiri Eğrisi")
                    st.line_chart(bt_result["equity"].rename("Strateji"))

                    st.subheader("Düşüş (Drawdown) Grafiği")
                    st.area_chart(bt_result["drawdown"])

                    st.subheader(f"İşlem Listesi ({len(bt_result['trades'])} işlem)")
                    if len(bt_result["trades"]) > 0:
                        st.dataframe(bt_result["trades"], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Bu dönemde hiç işlem üretilmedi.")

                    st.info(
                        "📌 **Nasıl okunmalı:** Kâr faktörü 1'in üzerindeyse, kazanan işlemlerin "
                        "toplamı kaybedenlerin toplamından fazla demektir. Az işlem sayısı ve büyük "
                        "maksimum düşüş, sonucun tesadüfe daha açık olabileceğine işaret eder. "
                        "Bu geçmiş performans, gelecekte aynı sonucu garanti etmez.",
                        icon="📌",
                    )

            except Exception as e:
                st.error(f"Hata oluştu: {e}")
    else:
        st.info("👆 Ayarları seçip **Backtest Çalıştır** butonuna bas.")

# --------------------------------------------------------------
# SEKME 4: POZİSYON BÜYÜKLÜĞÜ HESAPLAYICI
# --------------------------------------------------------------
with tab4:
    st.subheader("🧮 Pozisyon Büyüklüğü Hesaplayıcı")
    st.caption(
        "Sermayeni ve risk toleransını girerek, bir işlemde ne kadarlık pozisyon "
        "açmanın 'standart risk kuralına' uyacağını hesaplar. Bu bir tavsiye değil, "
        "sadece aritmetik bir araçtır."
    )

    with st.container(border=True):
        st.markdown("**1. Sermaye ve Risk**")
        pc1, pc2 = st.columns(2)
        with pc1:
            capital = st.number_input(
                "Toplam İşlem Sermayesi", min_value=0.0, value=100000.0, step=1000.0,
                help="Para birimi fark etmez — TL, USD, ne kullanıyorsan o birimde gir.",
            )
        with pc2:
            risk_pct = st.slider(
                "Bu İşlemde Riske Atılacak Sermaye (%)", 0.1, 10.0, 1.0, step=0.1,
                help="Yaygın bir kural: tek işlemde sermayenin %1-2'sinden fazlasını "
                     "riske atmamak. Bu senin tercihin, bot bir oran dayatmıyor.",
            )

    st.write("")

    with st.container(border=True):
        st.markdown("**2. İşlem Detayları**")
        direction = st.radio("Yön", ["LONG (Alış)", "SHORT (Satış)"], horizontal=True)
        is_long = direction.startswith("LONG")

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            entry_price = st.number_input("Giriş Fiyatı", min_value=0.0, value=2650.0, step=0.1)

        stop_method = st.radio(
            "Stop-Loss Nasıl Belirlensin?",
            ["Manuel fiyat gir", "ATR bazlı otomatik hesapla"],
            horizontal=True,
        )

        stop_price = None
        atr_info_text = ""

        if stop_method == "Manuel fiyat gir":
            with dc2:
                stop_price = st.number_input("Stop-Loss Fiyatı", min_value=0.0, value=2600.0, step=0.1)
            with dc3:
                take_profit = st.number_input("Kâr Al Fiyatı (opsiyonel)", min_value=0.0, value=0.0, step=0.1)
        else:
            atr_col1, atr_col2, atr_col3 = st.columns(3)
            with atr_col1:
                atr_symbol = st.selectbox(
                    "ATR için Enstrüman", list(GOLD_FAMILY.values()),
                    format_func=lambda x: next(k for k, v in GOLD_FAMILY.items() if v == x),
                    key="atr_symbol",
                )
            with atr_col2:
                atr_multiplier = st.number_input("ATR Çarpanı", min_value=0.5, value=2.0, step=0.5,
                                                   help="Yaygın kullanım: 1.5-3 arası. Yüksek çarpan = geniş stop.")
            with atr_col3:
                take_profit = st.number_input("Kâr Al Fiyatı (opsiyonel)", min_value=0.0, value=0.0, step=0.1)

            if atr_symbol:
                try:
                    with st.spinner(f"{atr_symbol} için ATR hesaplanıyor..."):
                        atr_raw_df = fetch_data(atr_symbol, period="6mo", interval="1d")
                        current_atr = float(atr(atr_raw_df, window=14).dropna().iloc[-1])
                    stop_distance_atr = current_atr * atr_multiplier
                    stop_price = (entry_price - stop_distance_atr) if is_long else (entry_price + stop_distance_atr)
                    atr_info_text = (
                        f"Güncel ATR(14): **{current_atr:.4f}** × çarpan {atr_multiplier} = "
                        f"stop mesafesi **{stop_distance_atr:.4f}** → hesaplanan stop: **{stop_price:.4f}**"
                    )
                except Exception as e:
                    st.error(f"ATR hesaplanamadı: {e}")

        if atr_info_text:
            st.info(atr_info_text)

    st.write("")

    calc_run = st.button("🧮 Hesapla", type="primary", use_container_width=True)

    if calc_run:
        if stop_price is None or stop_price == entry_price:
            st.error("Geçerli bir stop-loss fiyatı gerekiyor (giriş fiyatından farklı olmalı).")
        elif capital <= 0:
            st.error("Sermaye 0'dan büyük olmalı.")
        else:
            stop_distance = abs(entry_price - stop_price)
            risk_amount = capital * (risk_pct / 100)
            position_size = risk_amount / stop_distance
            position_value = position_size * entry_price
            leverage_needed = position_value / capital if capital > 0 else None

            st.divider()
            st.subheader("Sonuç")

            with st.container(border=True):
                r1, r2, r3 = st.columns(3)
                r1.metric("Risk Tutarı", f"{risk_amount:,.2f}")
                r2.metric("Önerilen Pozisyon Büyüklüğü", f"{position_size:,.4f} adet/lot")
                r3.metric("Toplam Pozisyon Değeri", f"{position_value:,.2f}")

                r4, r5 = st.columns(2)
                r4.metric("Stop-Loss Mesafesi", f"{stop_distance:.4f}")

                if take_profit and take_profit > 0:
                    reward = abs(take_profit - entry_price)
                    rr_ratio = reward / stop_distance if stop_distance > 0 else None
                    r5.metric("Risk / Ödül Oranı", f"1 : {rr_ratio:.2f}" if rr_ratio else "—")
                else:
                    r5.metric("Risk / Ödül Oranı", "Kâr al fiyatı girilmedi")

            if leverage_needed and leverage_needed > 1:
                st.warning(
                    f"⚠️ Bu pozisyon değeri ({position_value:,.2f}), toplam sermayeni "
                    f"({capital:,.2f}) aşıyor — yaklaşık **{leverage_needed:.2f}x kaldıraç** "
                    "gerektirir. Kaldıraçlı işlem riskini artırır, bunu bilerek ilerlediğinden emin ol.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"✅ Bu pozisyon, mevcut sermayenin içinde kaldıraçsız açılabilir "
                    f"(sermayenin ~%{(position_value/capital*100):.1f}'i kullanılıyor)."
                )

            st.caption(
                "📌 Formül: Risk Tutarı = Sermaye × Risk% · Pozisyon Büyüklüğü = Risk Tutarı ÷ "
                "Stop Mesafesi. Bu, yaygın kullanılan standart bir risk-bazlı pozisyon "
                "büyüklüğü hesaplama yöntemidir — kesin bir kural değil, bir başlangıç noktasıdır."
            )
    else:
        st.info("👆 Sermaye, risk yüzdesi ve işlem detaylarını gir, sonra **Hesapla** butonuna bas.")
