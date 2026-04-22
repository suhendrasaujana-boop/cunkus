import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from plotly.subplots import make_subplots

# ========== IMPORT OPSIONAL DENGAN PENANGANAN ERROR ==========
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ========== KONSTANTA ==========
TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
DEFAULT_TICKER = "^JKSE"
CACHE_TTL = 600
SCANNER_CACHE_TTL = 1800
IHSG_BLUE_CHIPS = [
    "BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "ADRO.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", "ICBP.JK",
    "INDF.JK", "UNVR.JK", "SMGR.JK", "CPIN.JK", "JPFA.JK"
]
VOLUME_SPIKE_THRESHOLD = 1.8
BREAKOUT_COOLDOWN_HOURS = 24
VOLUME_SPIKE_COOLDOWN_HOURS = 6

st.set_page_config(layout="wide", page_title="Smart Market Dashboard", page_icon="📊")

# ========== SESSION STATE ==========
if 'last_resistance' not in st.session_state:
    st.session_state.last_resistance = None
if 'last_breakout_notify_time' not in st.session_state:
    st.session_state.last_breakout_notify_time = None
if 'last_volume_ratio' not in st.session_state:
    st.session_state.last_volume_ratio = 0
if 'last_volume_notify_time' not in st.session_state:
    st.session_state.last_volume_notify_time = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'user_volume_spike_threshold' not in st.session_state:
    st.session_state.user_volume_spike_threshold = VOLUME_SPIKE_THRESHOLD
if 'user_breakout_cooldown_hours' not in st.session_state:
    st.session_state.user_breakout_cooldown_hours = BREAKOUT_COOLDOWN_HOURS

# ========== FUNGSI BANTU ==========
def safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^') or ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

def should_notify_breakout(current_price: float, resistance: float) -> bool:
    if current_price <= resistance:
        return False
    if st.session_state.last_resistance is None:
        return True
    if resistance > st.session_state.last_resistance:
        return True
    if st.session_state.last_breakout_notify_time is None:
        return True
    cooldown = timedelta(hours=st.session_state.user_breakout_cooldown_hours)
    if datetime.now() - st.session_state.last_breakout_notify_time > cooldown:
        return True
    return False

def should_notify_volume_spike(volume_ratio: float) -> bool:
    if volume_ratio <= st.session_state.user_volume_spike_threshold:
        return False
    if st.session_state.last_volume_notify_time is None:
        return True
    cooldown = timedelta(hours=VOLUME_SPIKE_COOLDOWN_HOURS)
    if datetime.now() - st.session_state.last_volume_notify_time > cooldown:
        return True
    return False

def show_notification(message: str, type: str = "toast", icon: str = "ℹ️"):
    if type == "toast" and hasattr(st, 'toast'):
        st.toast(message, icon=icon)
    else:
        if type == "success":
            st.success(message)
        elif type == "warning":
            st.warning(message)
        elif type == "info":
            st.info(message)
        else:
            st.info(message)

def get_fundamental_details(ticker: str) -> Dict[str, Any]:
    try:
        obj = yf.Ticker(ticker)
        info = obj.info
        return {
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'div_yield': info.get('dividendYield'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'roa': info.get('returnOnAssets'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margin': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
        }
    except Exception:
        return {}

@st.cache_data(ttl=CACHE_TTL)
def load_data(ticker: str, timeframe: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="2y", interval=timeframe, progress=False, auto_adjust=False)
        if df.empty:
            df = yf.download(ticker, period="1y", interval=timeframe, progress=False, auto_adjust=False)
        if df.empty or len(df) < 2:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how='all')
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=SCANNER_CACHE_TTL)
def scan_market_fast(tickers: List[str]) -> pd.DataFrame:
    try:
        time.sleep(random.uniform(0.3, 0.8))
        all_data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        rows = []
        for ticker in tickers:
            try:
                if isinstance(all_data.columns, pd.MultiIndex):
                    if ticker not in all_data.columns.levels[0]:
                        continue
                    df = all_data[ticker].dropna()
                else:
                    if ticker != all_data.columns.get_level_values(0)[0]:
                        continue
                    df = all_data.dropna()
                if len(df) < 20:
                    continue
                close = df['Close']
                rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) >= 14 else 50
                sma20 = close.rolling(20).mean().iloc[-1]
                if pd.isna(sma20):
                    sma20 = close.iloc[-1]
                score_val = (1 if rsi < 35 else 0) + (1 if close.iloc[-1] > sma20 else 0)
                rows.append([ticker, round(rsi, 2), score_val])
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["Ticker", "RSI", "Score"])
    except Exception as e:
        st.error(f"Scanner error: {e}")
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    df['SMA20'] = safe_sma(close, 20)
    df['SMA50'] = safe_sma(close, 50)

    if len(df) >= 14:
        df['RSI'] = ta.momentum.rsi(close, window=14)
    else:
        df['RSI'] = 50.0

    if len(df) >= 26:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
    else:
        df['MACD'] = 0.0
        df['MACD_signal'] = 0.0

    if volume.sum() > 0:
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0

    df['support'] = df['Low'].rolling(20, min_periods=1).min()
    df['resistance'] = df['High'].rolling(20, min_periods=1).max()

    try:
        df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
    except Exception:
        df['AD'] = 0.0
    try:
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
    except Exception:
        df['CMF'] = 0.0

    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
    else:
        df['BB_upper'] = np.nan
        df['BB_middle'] = np.nan
        df['BB_lower'] = np.nan

    df = df.ffill().fillna(0)
    return df

def calculate_ai_score(df: pd.DataFrame, volume: pd.Series) -> int:
    if df.empty:
        return 0
    conditions = [
        df['RSI'].iloc[-1] < 35,
        df['Close'].iloc[-1] > df['SMA20'].iloc[-1],
        df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1],
        df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1],
        volume.iloc[-1] > df['Volume_MA'].iloc[-1] if volume.sum() > 0 else False
    ]
    return sum(conditions)

def get_portfolio_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    if not tickers:
        return {}
    try:
        tickers_obj = yf.Tickers(' '.join(tickers))
        prices = {}
        for t in tickers:
            try:
                hist = tickers_obj.tickers[t].history(period="1d")
                if not hist.empty:
                    prices[t] = hist['Close'].iloc[-1]
                else:
                    prices[t] = None
            except Exception:
                prices[t] = None
        return prices
    except Exception as e:
        st.error(f"Error fetching portfolio prices: {e}")
        return {t: None for t in tickers}

def backtest_strategy(df, initial_capital=1000000, rsi_buy=35, rsi_sell=70, sma_period=20, commission=0.001):
    if df.empty or len(df) < 50:
        return None
    df = df.copy()
    df['sma'] = df['Close'].rolling(sma_period).mean()
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['buy_signal'] = (df['rsi'] < rsi_buy) & (df['Close'] > df['sma'])
    df['sell_signal'] = (df['rsi'] > rsi_sell) | (df['Close'] < df['sma'])
    
    position = 0
    cash = initial_capital
    trades = []
    equity_curve = [initial_capital]
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if df['buy_signal'].iloc[i] and position == 0:
            position = cash / price
            cash = 0
            trades.append(('BUY', df.index[i], price))
        elif df['sell_signal'].iloc[i] and position > 0:
            cash = position * price * (1 - commission)
            position = 0
            trades.append(('SELL', df.index[i], price))
        total_value = cash + (position * price if position > 0 else 0)
        equity_curve.append(total_value)
    
    if position > 0:
        cash = position * df['Close'].iloc[-1] * (1 - commission)
    final_capital = cash
    total_return = (final_capital - initial_capital) / initial_capital * 100
    bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'num_trades': len(trades),
        'trades': trades,
        'equity_curve': equity_curve,
        'bh_return': bh_return
    }

def prepare_ml_features(df, lookback=5, target_pct=0.01):
    if not SKLEARN_AVAILABLE:
        return None, None
    if df.empty or len(df) < 100:
        return None, None
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['ad'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
    df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
    df['momentum_5'] = df['Close'].pct_change(5)
    df['volume_change'] = df['Volume'] / df['Volume'].rolling(20).mean() - 1
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['future_return'] = df['Close'].shift(-lookback) / df['Close'] - 1
    df['target'] = (df['future_return'] > target_pct).astype(int)
    df = df.dropna()
    if len(df) < 50:
        return None, None
    features = ['rsi', 'sma20', 'sma50', 'volume_ma', 'ad', 'cmf', 'momentum_5', 'volume_change', 'atr']
    X = df[features]
    y = df['target']
    return X, y

def get_news_sentiment():
    if not (FEEDPARSER_AVAILABLE and TEXTBLOB_AVAILABLE):
        return None
    try:
        feed = feedparser.parse('https://www.cnbcindonesia.com/news/rss')
        sentiments = []
        for entry in feed.entries[:10]:
            blob = TextBlob(entry.title)
            sentiments.append(blob.sentiment.polarity)
        if sentiments:
            return sum(sentiments) / len(sentiments)
        return 0.0
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_multi_timeframe_trend(ticker):
    time.sleep(random.uniform(0.1, 0.3))
    daily = load_data(ticker, "1d")
    weekly = load_data(ticker, "1wk")
    monthly = load_data(ticker, "1mo")

    def trend(df):
        if df is None or len(df) < 20:
            return "Neutral"
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        if close.iloc[-1] > sma20.iloc[-1]:
            return "Bullish"
        elif close.iloc[-1] < sma20.iloc[-1]:
            return "Bearish"
        return "Neutral"

    return {
        "Daily": trend(daily),
        "Weekly": trend(weekly),
        "Monthly": trend(monthly)
    }

def calculate_smart_money(df):
    if df.empty or len(df) < 20:
        return 0, "Neutral"

    score = 0
    if df['CMF'].iloc[-1] > 0:
        score += 1
    if df['AD'].iloc[-1] > df['AD'].iloc[-5]:
        score += 1
    if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] * 1.5:
        score += 1
    if df['Close'].iloc[-1] > df['SMA20'].iloc[-1]:
        score += 1

    if score >= 3:
        status = "Accumulation"
    elif score == 2:
        status = "Neutral"
    else:
        status = "Distribution"

    return score, status

def get_macro_signal():
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False)['Close']
        usd = yf.download("USDIDR=X", period="5d", progress=False)['Close']
        nasdaq = yf.download("^IXIC", period="5d", progress=False)['Close']

        score = 0
        if ihsg.iloc[-1] > ihsg.mean():
            score += 1
        if usd.iloc[-1] < usd.mean():
            score += 1
        if nasdaq.iloc[-1] > nasdaq.mean():
            score += 1

        if score >= 2:
            return "Risk ON", score
        elif score == 1:
            return "Neutral", score
        else:
            return "Risk OFF", score
    except:
        return "Neutral", 1

@st.cache_data(ttl=3600)
def get_sector_rotation():
    sectors = {
        "Bank": ["BBRI.JK","BMRI.JK","BBCA.JK"],
        "Mining": ["ADRO.JK","ITMG.JK","ANTM.JK"],
        "Telco": ["TLKM.JK","EXCL.JK","ISAT.JK"],
        "Consumer": ["ICBP.JK","INDF.JK","UNVR.JK"]
    }

    sector_perf = {}
    for sector, tickers in sectors.items():
        returns = []
        for t in tickers:
            try:
                df = yf.download(t, period="5d", progress=False, auto_adjust=False)
                if df.empty:
                    continue
                close = df['Close'].squeeze()
                if len(close) < 2:
                    continue
                ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                returns.append(float(ret))
            except Exception:
                continue
        if returns:
            sector_perf[sector] = sum(returns) / len(returns)

    if not sector_perf:
        return "Neutral", sector_perf

    best = max(sector_perf, key=sector_perf.get)
    return best, sector_perf

def calculate_final_confidence(ai_score, smart_status, macro_status, best_sector, ticker):
    confidence = ai_score * 10
    if smart_status == "Accumulation":
        confidence += 15
    elif smart_status == "Distribution":
        confidence -= 15
    if macro_status == "Risk ON":
        confidence += 10
    elif macro_status == "Risk OFF":
        confidence -= 10

    sector_map = {
        "Bank": ["BBRI", "BMRI", "BBCA"],
        "Mining": ["ADRO", "ITMG", "ANTM"],
        "Telco": ["TLKM", "EXCL", "ISAT"],
        "Consumer": ["ICBP", "INDF", "UNVR"]
    }

    for sector, stocks in sector_map.items():
        if ticker.replace(".JK","") in stocks and sector == best_sector:
            confidence += 5

    confidence = max(0, min(100, confidence))
    return confidence

def get_all_signals(df, volume, ticker):
    ai_score = calculate_ai_score(df, volume)

    smart_score, smart_status = calculate_smart_money(df)

    macro_status, macro_score = get_macro_signal()

    best_sector, _ = get_sector_rotation()

    confidence = calculate_final_confidence(
        ai_score,
        smart_status,
        macro_status,
        best_sector,
        ticker
    )

    return {
        'ai_score': ai_score,
        'smart_score': smart_score,
        'smart_status': smart_status,
        'macro_status': macro_status,
        'macro_score': macro_score,
        'best_sector': best_sector,
        'confidence': confidence
    }
    
def weighted_decision_engine(df, volume, ticker):
    signals = get_all_signals(df, volume, ticker)
    # technical
    tech_score = signals['ai_score'] / 5

    # smart money
    smart_map = {
        "Accumulation": 1,
        "Neutral": 0.5,
        "Distribution": 0
    }
    smart_score = smart_map.get(signals['smart_status'], 0.5)

    # macro
    macro_map = {
        "Risk ON": 1,
        "Neutral": 0.5,
        "Risk OFF": 0
    }
    macro_score = macro_map.get(signals['macro_status'], 0.5)

    # sector
    sector_score = 1 if ticker.replace(".JK","") in signals['best_sector'] else 0.5

    # risk
    returns = df['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    if vol < 0.15:
        risk_score = 1
    elif vol < 0.30:
        risk_score = 0.6
    else:
        risk_score = 0.2

    # momentum
    momentum = df['Close'].pct_change(5).iloc[-1]
    if momentum > 0.05:
        momentum_score = 1
    elif momentum > 0:
        momentum_score = 0.6
    else:
        momentum_score = 0.2

    final_score = (
        0.30 * tech_score +
        0.20 * smart_score +
        0.15 * macro_score +
        0.10 * sector_score +
        0.10 * risk_score +
        0.15 * momentum_score
    )

    return round(final_score * 100, 2)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# 📊 Smart Market Dashboard")
    ticker_input = st.text_input("Ticker", DEFAULT_TICKER, help="Contoh: BBCA, BBRI, ASII, atau ^JKSE untuk IHSG.")
    ticker = fix_ticker(ticker_input)
    if ticker != ticker_input:
        st.info(f"Format: {ticker}")
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    
    st.markdown("### ⚙️ Pengaturan Alert")
    user_volume_spike = st.slider("Volume spike multiplier", 1.0, 5.0, 
                                  st.session_state.user_volume_spike_threshold, 0.1,
                                  key="user_volume_spike_slider")
    user_breakout_cooldown = st.number_input("Breakout cooldown (jam)", 1, 72, 
                                             st.session_state.user_breakout_cooldown_hours,
                                             key="user_breakout_cooldown_input")
    st.session_state.user_volume_spike_threshold = user_volume_spike
    st.session_state.user_breakout_cooldown_hours = user_breakout_cooldown
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Data dari Yahoo Finance | Update 10 menit")

# ========== LOAD DATA ==========
with st.spinner("Memuat data..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 5:
    st.warning(f"Data tidak cukup untuk {ticker} (minimal 5 periode).")
    st.stop()

data = add_indicators(data)

if not data.empty:
    current_resistance = data['resistance'].iloc[-1]
    current_price = data['Close'].iloc[-1]
    if should_notify_breakout(current_price, current_resistance):
        show_notification(f"🚀 BREAKOUT! Harga menembus resistance {current_resistance:.2f}", "toast", "🚀")
        st.session_state.last_resistance = current_resistance
        st.session_state.last_breakout_notify_time = datetime.now()
    
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
    if volume.sum() > 0:
        vol_last = volume.iloc[-1]
        vol_ma = data['Volume_MA'].iloc[-1]
        if vol_ma > 0:
            volume_ratio = vol_last / vol_ma
            if should_notify_volume_spike(volume_ratio):
                show_notification(f"🔥 Volume Spike! {volume_ratio:.1f}x", "toast", "⚠️")
                st.session_state.last_volume_ratio = volume_ratio
                st.session_state.last_volume_notify_time = datetime.now()

# ========== HEADER HARGA ==========
st.title(f"📈 {ticker}")
last_close = data['Close'].iloc[-1]
last_high = data['High'].iloc[-1]
last_low = data['Low'].iloc[-1]
prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Terakhir", f"{last_close:.2f}", f"{change_pct:.2f}%", delta_color="normal")
col2.metric("Hari Ini - Tertinggi", f"{last_high:.2f}")
col3.metric("Hari Ini - Terendah", f"{last_low:.2f}")
col4.metric("Volume Terakhir", f"{volume.iloc[-1]:,.0f}" if volume.sum() > 0 else "N/A")

ad_val = data['AD'].iloc[-1]
cmf_val = data['CMF'].iloc[-1]
ad_status = "Akumulasi" if ad_val > 0 else "Distribusi" if ad_val < 0 else "Netral"
cmf_status = "Akumulasi" if cmf_val > 0 else "Distribusi" if cmf_val < 0 else "Netral"
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <b>📊 Akumulasi/Distribusi (AD):</b> {ad_status} ({ad_val:.2f}) &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>💰 Chaikin Money Flow (CMF20):</b> {cmf_status} ({cmf_val:.3f})
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Grafik", "🤖 AI Signal", "🔍 Scanner", "📁 Portfolio",
    "🧪 Backtest & ML", "📖 Info", "🏦 Smart Money", "🌍 Macro Market", "🔄 Sector Flow"
])

# ========== TAB 1: GRAFIK ==========
with tab1:
    st.subheader("Candlestick Chart dengan Volume")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name="Price",
                                 hovertemplate='Tanggal: %{x}<br>Open: %{open:.2f}<br>High: %{high:.2f}<br>Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA20",
                             hovertemplate='SMA20: %{y:.2f}<extra></extra>'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA50",
                             hovertemplate='SMA50: %{y:.2f}<extra></extra>'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['support'], name="Support", line=dict(dash='dash'),
                             hovertemplate='Support: %{y:.2f}<extra></extra>'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['resistance'], name="Resistance", line=dict(dash='dash'),
                             hovertemplate='Resistance: %{y:.2f}<extra></extra>'),
                  row=1, col=1)
    if 'BB_upper' in data.columns and not data['BB_upper'].isnull().all():
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name="BB Upper", line=dict(dash='dot'),
                                 hovertemplate='BB Upper: %{y:.2f}<extra></extra>'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name="BB Lower", line=dict(dash='dot'),
                                 hovertemplate='BB Lower: %{y:.2f}<extra></extra>'),
                      row=1, col=1)
    
    fig.add_trace(go.Bar(x=data.index, y=volume, name="Volume", marker_color='lightblue',
                         hovertemplate='Volume: %{y:,.0f}<extra></extra>'),
                  row=2, col=1)
    
    fig.update_layout(height=800, title_text=f"{ticker} - Candlestick & Volume", showlegend=True)
    fig.update_xaxes(title_text="Tanggal", row=2, col=1)
    fig.update_yaxes(title_text="Harga", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI (14) & MACD")
    col_r, col_m = st.columns(2)
    with col_r:
        st.line_chart(data['RSI'])
        st.caption("RSI < 35: Oversold | >70: Overbought")
    with col_m:
        st.line_chart(data[['MACD', 'MACD_signal']])
        st.caption("MACD > Signal: Bullish")

    st.subheader("AD & CMF")
    col_ad, col_cmf = st.columns(2)
    with col_ad:
        st.line_chart(data['AD'])
    with col_cmf:
        st.line_chart(data['CMF'])

# ========== TAB 2: AI SIGNAL ==========
with tab2:
    all_signals = get_all_signals(data, volume, ticker)
    score = all_signals['ai_score']
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🤖 AI Score")
        st.metric("Score", f"{score}/5")
        if score >= 4:
            st.success("🚀 STRONG BUY")
        elif score == 3:
            st.info("🟡 HOLD")
        else:
            st.error("🔻 SELL")
    with col_b:
        st.subheader("🎯 Risk Meter")
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            if timeframe == "1d":
                periods_per_year = 252
            elif timeframe == "1wk":
                periods_per_year = 52
            else:
                periods_per_year = 12
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(periods_per_year) * 100
        else:
            annual_vol = 0.0
        if annual_vol < 15:
            risk = "LOW"
            st.success(f"Risk Level: {risk} ({annual_vol:.1f}% annual)")
        elif annual_vol < 30:
            risk = "MEDIUM"
            st.warning(f"Risk Level: {risk} ({annual_vol:.1f}% annual)")
        else:
            risk = "HIGH"
            st.error(f"Risk Level: {risk} ({annual_vol:.1f}% annual)")

    st.subheader("📊 Probability Engine")
    bull = bear = 0
    if data['RSI'].iloc[-1] < 35:
        bull += 1
    else:
        bear += 1
    if data['Close'].iloc[-1] > data['SMA20'].iloc[-1]:
        bull += 1
    else:
        bear += 1
    if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        bull += 1
    else:
        bear += 1
    total = bull + bear
    if total > 0:
        bull_prob = (bull/total)*100
        bear_prob = (bear/total)*100
    else:
        bull_prob = bear_prob = 50
    st.write(f"📈 Bullish: {bull_prob:.1f}% | 📉 Bearish: {bear_prob:.1f}%")

    st.subheader("🔥 Momentum Detector")
    if len(data) >= 6:
        momentum = data['Close'].pct_change(5).iloc[-1] * 100
        if pd.isna(momentum):
            momentum = 0
    else:
        momentum = 0
    if momentum > 5:
        st.success("Strong Up Momentum")
    elif momentum > 2:
        st.info("Moderate Up Momentum")
    elif momentum < -5:
        st.error("Strong Down Momentum")
    else:
        st.warning("Sideways")

    col_brk, col_vol = st.columns(2)
    with col_brk:
        st.subheader("🚀 Breakout Detector")
        if len(data) > 1 and data['Close'].iloc[-1] > data['resistance'].iloc[-2]:
            st.success("🔥 BREAKOUT DETECTED")
        else:
            st.info("Tidak ada breakout")
    with col_vol:
        st.subheader("🚨 Volume Alert")
        if volume.sum() > 0:
            if volume.iloc[-1] > data['Volume_MA'].iloc[-1] * st.session_state.user_volume_spike_threshold:
                st.success("Volume Spike Detected")
            else:
                st.write("Normal Volume")
        else:
            st.info("Volume tidak tersedia")

    st.header("🧠 AI FINAL PRO")
    final_score = 0
    if bull_prob > 60:
        final_score += 1
    if risk == "LOW":
        final_score += 1
    if momentum > 2:
        final_score += 1
    if volume.sum() > 0 and volume.iloc[-1] > data['Volume_MA'].iloc[-1]:
        final_score += 1
    if final_score >= 2:
        st.success("🚀 STRONG ACCUMULATE")
    elif final_score == 1:
        st.warning("🟡 SPEC BUY")
    else:
        st.error("🔻 WAIT / AVOID")

    st.header("🚀 GOD MODE TRADING ENGINE")
    price = data['Close'].iloc[-1]
    support = data['support'].iloc[-1] if pd.notna(data['support'].iloc[-1]) else price * 0.95
    resistance = data['resistance'].iloc[-1] if pd.notna(data['resistance'].iloc[-1]) else price * 1.05

    entry = (support + price) / 2
    stoploss = support * 0.97
    target1 = resistance
    risk_amount = entry - stoploss
    reward = target1 - entry
    rr = reward / risk_amount if risk_amount > 0 else 0

    col_e, col_s, col_t = st.columns(3)
    col_e.metric("🎯 Smart Entry", f"{entry:.2f}")
    col_s.metric("🛑 Stoploss", f"{stoploss:.2f}")
    col_t.metric("💰 Target 1", f"{target1:.2f}")
    st.write(f"**Risk Reward Ratio:** 1 : {rr:.2f}")
    if rr > 2:
        st.success("Good Trade Setup")
    elif rr > 1:
        st.warning("Moderate Setup")
    else:
        st.error("Bad Setup")

    col_ts, col_bs = st.columns(2)
    with col_ts:
        st.subheader("📈 Trend Score")
        trend_score = 0
        if price > data['SMA20'].iloc[-1]:
            trend_score += 1
        if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1]:
            trend_score += 1
        if data['RSI'].iloc[-1] > 50:
            trend_score += 1
        st.write(f"Trend Score: {trend_score}/3")
    with col_bs:
        st.subheader("🔥 Breakout Strength")
        if price > resistance:
            strength = (price - resistance) / resistance * 100
            st.success(f"Breakout Strength: {strength:.2f}%")
        else:
            st.write("No Breakout")

    st.header("🧠 FINAL GOD SIGNAL")
    god_score = 0
    if rr > 2:
        god_score += 1
    if trend_score >= 2:
        god_score += 1
    if data['RSI'].iloc[-1] < 70:
        god_score += 1
    if price > data['SMA20'].iloc[-1]:
        god_score += 1
    if god_score >= 3:
        st.success("🚀 GOD MODE BUY")
    elif god_score == 2:
        st.warning("⚡ SPEC BUY")
    else:
        st.error("❌ NO TRADE")

    st.header("FINAL DECISION")
    if score >= 3:
        st.success("🟢 ACCUMULATE")
    elif score == 2:
        st.warning("🟡 WAIT")
    else:
        st.error("🔴 AVOID")

    st.subheader("🧭 Multi Timeframe Confirmation")
    mtf = get_multi_timeframe_trend(ticker)
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily", mtf["Daily"])
    col2.metric("Weekly", mtf["Weekly"])
    col3.metric("Monthly", mtf["Monthly"])
    bull_count = list(mtf.values()).count("Bullish")
    if bull_count == 3:
        st.success("🚀 Strong Multi-Timeframe Bullish")
    elif bull_count == 2:
        st.info("🟡 Moderate Bullish")
    else:
        st.warning("⚠️ Weak Trend")

    st.subheader("🏦 Smart Money Flow")
    smart_score, smart_status = all_signals['smart_score'], all_signals['smart_status']
    col1, col2 = st.columns(2)
    col1.metric("Smart Money Score", f"{smart_score}/4")
    col2.metric("Status", smart_status)
    if smart_status == "Accumulation":
        st.success("💰 Smart Money Accumulating")
    elif smart_status == "Distribution":
        st.error("⚠️ Smart Money Distributing")
    else:
        st.info("Sideways / Neutral")

    st.subheader("🌍 Macro Market Filter")
    macro_status, macro_score = all_signals['macro_status'], all_signals['macro_score']
    col1, col2 = st.columns(2)
    col1.metric("Macro Condition", macro_status)
    col2.metric("Score", f"{macro_score}/3")
    if macro_status == "Risk ON":
        st.success("📈 Market Supportive")
    elif macro_status == "Risk OFF":
        st.error("⚠️ Market Risk-Off")
    else:
        st.info("Market Neutral")

    st.subheader("🔄 Sector Rotation")
    best_sector = all_signals['best_sector']
    st.metric("Leading Sector", best_sector)
    _, sector_data = get_sector_rotation()
    for sector, val in sector_data.items():
        st.write(f"{sector}: {val:.2%}")

    st.subheader("🎯 Final AI Confidence")
    confidence = all_signals['confidence']
    st.metric("Confidence Score", f"{confidence}%")
    if confidence >= 75:
        st.success("🚀 High Probability Trade")
    elif confidence >= 60:
        st.info("👍 Moderate Probability")
    else:
        st.warning("⚠️ Low Confidence")

    st.subheader("🧠 Weighted Decision Engine (PRO)")
    weighted_score = weighted_decision_engine(data, volume, ticker)
    st.metric("Weighted Score", f"{weighted_score}%")
    if weighted_score >= 75:
        st.success("🚀 STRONG BUY (High Confidence)")
    elif weighted_score >= 60:
        st.info("👍 BUY (Moderate)")
    elif weighted_score >= 45:
        st.warning("🟡 HOLD / WAIT")
    else:
        st.error("🔻 AVOID")

# ========== TAB 3: SCANNER ==========
with tab3:
    st.subheader("🔥 SUPER FAST IHSG SCANNER (15 Blue Chip)")
    with st.spinner("Memindai 15 saham..."):
        scan_df = scan_market_fast(IHSG_BLUE_CHIPS)
    if not scan_df.empty:
        st.dataframe(scan_df.sort_values("Score", ascending=False), use_container_width=True)
        st.subheader("🚀 TOP 5 BUY")
        top5 = scan_df.sort_values("Score", ascending=False).head(5)
        if not top5.empty:
            st.table(top5)
        else:
            st.info("Tidak ada saham dengan score > 0")
    else:
        st.warning("Scanner gagal, coba lagi nanti.")

# ========== TAB 4: PORTFOLIO ==========
with tab4:
    st.subheader("📁 Portfolio Tracker")
    with st.expander("➕ Tambah Posisi Baru"):
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            ticker_entry = st.text_input("Ticker", key="entry_ticker", help="Contoh: BBCA.JK")
        with col_t2:
            entry_date = st.date_input("Tanggal Entry", value=date.today())
        with col_t3:
            entry_price = st.number_input("Harga Entry", min_value=0.0, step=10.0)
        with col_t4:
            shares = st.number_input("Jumlah Saham", min_value=1, step=100)
        if st.button("Simpan Posisi"):
            if ticker_entry and entry_price > 0 and shares > 0:
                st.session_state.portfolio.append({
                    'ticker': fix_ticker(ticker_entry),
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'shares': shares
                })
                st.success("Posisi ditambahkan!")
                st.rerun()
            else:
                st.error("Isi semua field dengan benar.")
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        unique_tickers = portfolio_df['ticker'].unique().tolist()
        current_prices = get_portfolio_current_prices(unique_tickers)
        portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
        portfolio_df['unrealized_pnl'] = (portfolio_df['current_price'] - portfolio_df['entry_price']) * portfolio_df['shares']
        portfolio_df['pnl_pct'] = ((portfolio_df['current_price'] - portfolio_df['entry_price']) / portfolio_df['entry_price']) * 100
        st.dataframe(portfolio_df, use_container_width=True)
        total_pnl = portfolio_df['unrealized_pnl'].sum()
        st.metric("Total Unrealized P&L", f"{total_pnl:,.2f}", delta=f"{total_pnl:+,.2f}")
        if st.button("Hapus Semua Posisi"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Belum ada posisi. Gunakan form di atas untuk menambahkan.")
    st.divider()
    st.subheader("📐 Position Sizing Calculator")
    col_cap, col_risk, col_sl = st.columns(3)
    with col_cap:
        capital = st.number_input("Modal (Rp)", min_value=0.0, value=100_000_000.0, step=10_000_000.0)
    with col_risk:
        risk_percent = st.number_input("Risiko per Trade (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    with col_sl:
        stoploss_price = st.number_input("Stop Loss (Rp)", min_value=0.0, value=last_close * 0.97 if last_close else 0.0)
    if capital > 0 and risk_percent > 0 and stoploss_price > 0 and last_close > 0:
        risk_amount = capital * (risk_percent / 100)
        price_risk = last_close - stoploss_price
        if price_risk > 0:
            suggested_shares = int(risk_amount / price_risk)
            position_value = suggested_shares * last_close
            st.write(f"**Jumlah saham yang direkomendasikan:** {suggested_shares:,} lembar")
            st.write(f"Nilai posisi: Rp {position_value:,.2f} ({position_value/capital*100:.1f}% dari modal)")
            if position_value > capital:
                st.warning("Nilai posisi melebihi modal! Turunkan jumlah saham atau perbesar stop loss.")
        else:
            st.warning("Stop loss harus di bawah harga saat ini.")
    else:
        st.info("Masukkan modal, risiko, dan stop loss untuk menghitung.")

# ========== TAB 5: BACKTEST & ML ==========
with tab5:
    st.header("🧪 Backtesting & Machine Learning")
    
    if volume.sum() == 0:
        st.warning("Data volume tidak tersedia – backtest dan ML akan menggunakan fitur terbatas.")
    
    st.subheader("📈 Backtest Strategi (RSI + SMA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_buy = st.slider("RSI Buy Threshold", min_value=20, max_value=45, value=35, step=1)
    with col2:
        rsi_sell = st.slider("RSI Sell Threshold", min_value=60, max_value=85, value=70, step=1)
    with col3:
        sma_period = st.slider("SMA Period", min_value=10, max_value=50, value=20, step=5)
    
    commission = st.checkbox("Include Commission (0.1% per trade)", value=True)
    commission_rate = 0.001 if commission else 0
    
    with st.spinner("Menjalankan backtest..."):
        bt_result = backtest_strategy(data, rsi_buy=rsi_buy, rsi_sell=rsi_sell, sma_period=sma_period, commission=commission_rate)
    
    if bt_result:
        col_ret, col_trades, col_bh = st.columns(3)
        col_ret.metric("Total Return", f"{bt_result['total_return']:.2f}%", delta=f"{bt_result['total_return']:.2f}%")
        col_trades.metric("Jumlah Transaksi", bt_result['num_trades'])
        col_bh.metric("Buy & Hold Return", f"{bt_result['bh_return']:.2f}%", delta=f"{bt_result['bh_return']:.2f}%")
        st.write(f"**Modal Akhir:** Rp {bt_result['final_capital']:,.2f}")
        
        st.subheader("📈 Equity Curve")
        equity_df = pd.DataFrame({'Date': data.index[:len(bt_result['equity_curve'])], 'Equity': bt_result['equity_curve']})
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Equity'], mode='lines', name='Equity'))
        fig_eq.update_layout(title="Pertumbuhan Modal", xaxis_title="Tanggal", yaxis_title="Nilai (Rp)")
        st.plotly_chart(fig_eq, use_container_width=True)
        
        with st.expander("Lihat Detail Transaksi"):
            if bt_result['trades']:
                st.dataframe(pd.DataFrame(bt_result['trades'], columns=['Tipe', 'Tanggal', 'Harga']))
            else:
                st.info("Tidak ada transaksi.")
    else:
        st.warning(f"Data tidak cukup untuk backtest. Minimal 50 baris, saat ini {len(data)} baris.")
    
    st.divider()
    
    st.subheader("🤖 Machine Learning Prediction (Walk-Forward Validation)")
    
    if SKLEARN_AVAILABLE:
        lookback = st.slider("Prediction Horizon (hari)", min_value=1, max_value=10, value=5, step=1)
        target_pct = st.slider("Target Return (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5) / 100.0
        
        X, y = prepare_ml_features(data, lookback=lookback, target_pct=target_pct)
        
        if X is not None and len(X) > 100:
            window_size = 200
            accuracies = []
            last_model = None
            for i in range(window_size, len(X) - 50, 50):
                train_X = X.iloc[i-window_size:i]
                train_y = y.iloc[i-window_size:i]
                test_X = X.iloc[i:i+50]
                test_y = y.iloc[i:i+50]
                if len(train_X) < 100 or len(test_X) < 10:
                    continue
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(train_X, train_y)
                pred = model.predict(test_X)
                acc = accuracy_score(test_y, pred)
                accuracies.append(acc)
                last_model = model
            
            avg_acc = np.mean(accuracies) if accuracies else 0
            st.metric("Average Walk-Forward Accuracy", f"{avg_acc:.2%}")
            st.caption("Akurasi rata-rata dari rolling window validasi.")
            
            if last_model is not None:
                latest = X.iloc[-1:].values
                prob = last_model.predict_proba(latest)[0][1]
                st.metric(f"Probabilitas harga naik >{target_pct*100:.1f}% dalam {lookback} hari", f"{prob:.2%}")
            else:
                st.info("Belum ada model yang terbentuk, coba gunakan data yang lebih panjang.")
            
            if st.checkbox("Tampilkan Feature Importance") and last_model is not None:
                importance = pd.DataFrame({'Feature': X.columns, 'Importance': last_model.feature_importances_})
                st.bar_chart(importance.set_index('Feature'))
        else:
            if len(data) < 200:
                st.warning(f"Data tidak cukup untuk ML. Minimal 200 baris, saat ini {len(data)} baris.")
            else:
                st.warning("Data tidak cukup setelah preprocessing. Pastikan volume tersedia (saham biasa, bukan indeks).")
    else:
        st.error("❌ Machine learning tidak tersedia karena library 'scikit-learn' tidak terinstal. Install dengan: pip install scikit-learn")
    
    st.divider()
    
    st.subheader("📰 Sentimen Berita (Diversifikasi)")
    if FEEDPARSER_AVAILABLE and TEXTBLOB_AVAILABLE:
        with st.spinner("Mengambil sentimen berita..."):
            sentiment = get_news_sentiment()
        if sentiment is not None:
            sentiment_text = "Positif" if sentiment > 0 else "Negatif" if sentiment < 0 else "Netral"
            st.metric("Sentimen 24 Jam", f"{sentiment_text} ({sentiment:.2f})")
            st.caption("Sumber: CNBC Indonesia RSS (judul berita terbaru)")
        else:
            st.info("Tidak dapat mengambil sentimen berita saat ini. Cek koneksi internet atau RSS feed.")
    else:
        missing = []
        if not FEEDPARSER_AVAILABLE:
            missing.append("feedparser")
        if not TEXTBLOB_AVAILABLE:
            missing.append("textblob")
        st.error(f"❌ Fitur sentimen tidak tersedia karena library {', '.join(missing)} tidak terinstal. Install dengan: pip install {' '.join(missing)}")

# ========== TAB 6: INFO ==========
with tab6:
    if not ticker.startswith('^'):
        st.subheader("📊 Fundamental Details")
        fund = get_fundamental_details(ticker)
        if fund:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PER (TTM)", f"{fund['pe']:.2f}" if pd.notna(fund['pe']) else "N/A")
                st.metric("PBV", f"{fund['pb']:.2f}" if pd.notna(fund['pb']) else "N/A")
                st.metric("Dividend Yield", f"{fund['div_yield']*100:.2f}%" if pd.notna(fund['div_yield']) else "N/A")
                st.metric("Market Cap", f"{fund['market_cap']/1e12:.2f}T" if pd.notna(fund['market_cap']) else "N/A")
                st.metric("Sektor", fund['sector'] if fund['sector'] else "N/A")
            with col2:
                st.metric("ROA", f"{fund['roa']*100:.2f}%" if pd.notna(fund['roa']) else "N/A")
                st.metric("ROE", f"{fund['roe']*100:.2f}%" if pd.notna(fund['roe']) else "N/A")
                st.metric("Debt to Equity", f"{fund['debt_to_equity']:.2f}" if pd.notna(fund['debt_to_equity']) else "N/A")
                st.metric("Profit Margin", f"{fund['profit_margin']*100:.2f}%" if pd.notna(fund['profit_margin']) else "N/A")
                st.metric("Revenue Growth (YoY)", f"{fund['revenue_growth']*100:.2f}%" if pd.notna(fund['revenue_growth']) else "N/A")
        else:
            st.info("Data fundamental tidak tersedia untuk ticker ini.")
    else:
        st.info("Data fundamental tidak tersedia untuk indeks.")
    
    st.divider()
    
    if FEEDPARSER_AVAILABLE and TEXTBLOB_AVAILABLE:
        st.subheader("📰 Diversifikasi Sinyal: Sentimen Berita Terkini")
        sentiment = get_news_sentiment()
        if sentiment is not None:
            sentiment_text = "Positif" if sentiment > 0 else "Negatif" if sentiment < 0 else "Netral"
            st.metric("Sentimen 24 Jam", f"{sentiment_text} ({sentiment:.2f})")
            st.caption("Sumber: CNBC Indonesia RSS (judul berita terbaru)")
        else:
            st.info("Tidak dapat mengambil sentimen saat ini.")
    
    with st.expander("📖 Glossary (Klik untuk lihat)"):
        st.markdown("""
        **RSI** < 35: Oversold | >70: Overbought  
        **SMA20/50**: Harga > SMA = uptrend  
        **MACD > Signal**: Bullish  
        **Volume > Volume MA**: Volume di atas rata-rata  
        **AD > 0**: Akumulasi (tekanan beli)  
        **CMF > 0**: Tekanan beli  
        **Risk Reward Ratio**: Target/risk > 2 = good setup  
        **AI Score** 4-5: Strong Buy, 3: Hold, 0-2: Sell  
        **FINAL DECISION** score ≥3: Accumulate, =2: Wait, ≤1: Avoid
        **Backtest**: Menguji strategi RSI + SMA pada data historis, sekarang dengan parameter custom dan komisi
        **Machine Learning**: Prediksi probabilitas kenaikan harga dengan fitur momentum, volume change, ATR dan validasi walk-forward
        **Multi Timeframe**: Bullish jika harga > SMA20 di daily, weekly, monthly
        **Smart Money**: Akumulasi jika CMF>0, AD naik, volume spike, price>MA20
        **Macro**: Risk ON jika IHSG naik, USD turun, Nasdaq naik
        **Sector Rotation**: Sektor dengan performa 5 hari terbaik
        **Final Confidence**: Gabungan AI score + Smart Money + Macro + Sector
        """)

# ========== TAB 7: SMART MONEY ==========
with tab7:
    st.header("🏦 Smart Money / Bandarmology Proxy")
    st.markdown("Deteksi akumulasi/distribusi berdasarkan volume dan harga.")
    if not data.empty:
        smart_score, smart_status = calculate_smart_money(data)
        st.metric("Smart Money Score", f"{smart_score}/4")
        st.metric("Status", smart_status)
        if smart_status == "Accumulation":
            st.success("💰 Smart Money sedang mengakumulasi (tekanan beli).")
        elif smart_status == "Distribution":
            st.error("⚠️ Smart Money mendistribusikan (tekanan jual).")
        else:
            st.info("Netral, belum ada sinyal kuat.")
        
        st.subheader("Komponen Score")
        cmf_ok = data['CMF'].iloc[-1] > 0
        ad_ok = data['AD'].iloc[-1] > data['AD'].iloc[-5]
        vol_ok = data['Volume'].iloc[-1] > data['Volume_MA'].iloc[-1] * 1.5
        price_ok = data['Close'].iloc[-1] > data['SMA20'].iloc[-1]
        st.write(f"✅ CMF > 0: {cmf_ok}")
        st.write(f"✅ AD Naik (5 hari): {ad_ok}")
        st.write(f"✅ Volume Spike (>1.5x MA): {vol_ok}")
        st.write(f"✅ Harga > SMA20: {price_ok}")
    else:
        st.warning("Data tidak cukup.")

# ========== TAB 8: MACRO MARKET ==========
with tab8:
    st.header("🌍 Macro Market Dashboard")
    st.markdown("Kondisi makro global untuk filter keputusan.")
    macro_status, macro_score = get_macro_signal()
    st.metric("Macro Condition", macro_status)
    st.metric("Macro Score", f"{macro_score}/3")
    if macro_status == "Risk ON":
        st.success("📈 Kondisi mendukung risk-on. Peluang lebih tinggi.")
    elif macro_status == "Risk OFF":
        st.error("⚠️ Kondisi risk-off. Hindari agresif.")
    else:
        st.info("Netral.")
    
    try:
        ihsg = yf.download("^JKSE", period="5d", progress=False)['Close']
        usd = yf.download("USDIDR=X", period="5d", progress=False)['Close']
        nasdaq = yf.download("^IXIC", period="5d", progress=False)['Close']
        st.subheader("Data Terkini")
        col1, col2, col3 = st.columns(3)
        col1.metric("IHSG", f"{ihsg.iloc[-1]:.2f}", f"{((ihsg.iloc[-1]/ihsg.iloc[0])-1)*100:.2f}%")
        col2.metric("USD/IDR", f"{usd.iloc[-1]:.2f}", f"{((usd.iloc[-1]/usd.iloc[0])-1)*100:.2f}%")
        col3.metric("Nasdaq", f"{nasdaq.iloc[-1]:.2f}", f"{((nasdaq.iloc[-1]/nasdaq.iloc[0])-1)*100:.2f}%")
    except:
        st.info("Tidak dapat mengambil data makro saat ini.")

# ========== TAB 9: SECTOR FLOW ==========
with tab9:
    st.header("🔄 Sector Rotation")
    st.markdown("Sektor dengan performa terbaik 5 hari terakhir.")
    best, sector_data = get_sector_rotation()
    st.metric("Leading Sector", best)
    if sector_data:
        df_sector = pd.DataFrame(list(sector_data.items()), columns=["Sektor", "Return 5 Hari"])
        df_sector = df_sector.sort_values("Return 5 Hari", ascending=False)
        st.bar_chart(df_sector.set_index("Sektor"))
    else:
        st.info("Tidak dapat mengambil data sektor.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **DISCLAIMER:** Dashboard ini hanya untuk edukasi dan analisis otomatis. Bukan rekomendasi beli/jual. Keputusan investasi sepenuhnya risiko Anda.")
