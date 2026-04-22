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
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== TAMBAHAN UNTUK PORTFOLIO OPTIMIZER ==========
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ========== DETEKSI ENVIRONMENT ==========
IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE', '').lower() == 'sharing'

# ========== FITUR YANG DINONAKTIFKAN DI CLOUD ==========
ENABLE_ML = True
ENABLE_NEWS = False if IS_CLOUD else True
ENABLE_SENTIMENT = False if IS_CLOUD else True
ENABLE_MULTI_TICKER_ML = not IS_CLOUD

# ========== IMPORT OPSIONAL ==========
FEEDPARSER_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
if ENABLE_NEWS:
    try:
        import feedparser
        FEEDPARSER_AVAILABLE = True
    except ImportError:
        pass
if ENABLE_SENTIMENT:
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
    except ImportError:
        pass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ========== KONSTANTA ==========
if IS_CLOUD:
    DATA_PERIOD = "6mo"
    MAX_CHART_POINTS = 150
    CACHE_TTL = 300
    SCANNER_CACHE_TTL = 900
    st.warning("☁️ **Mode Cloud Aktif** - Optimasi untuk performa terbaik. Multi-ticker ML dinonaktifkan, ensemble tetap jalan.")
else:
    DATA_PERIOD = "2y"
    MAX_CHART_POINTS = 500
    CACHE_TTL = 600
    SCANNER_CACHE_TTL = 1800

TIMEFRAMES = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
DEFAULT_TICKER = "^JKSE"
IHSG_BLUE_CHIPS = [
    "BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "ADRO.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", "ICBP.JK",
    "INDF.JK", "UNVR.JK", "SMGR.JK", "CPIN.JK", "JPFA.JK"
]
VOLUME_SPIKE_THRESHOLD = 1.8
BREAKOUT_COOLDOWN_HOURS = 24
VOLUME_SPIKE_COOLDOWN_HOURS = 6

st.set_page_config(
    layout="wide", 
    page_title="Smart Market Dashboard", 
    page_icon="📊",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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
if 'global_ml_model' not in st.session_state:
    st.session_state.global_ml_model = None
if 'global_feature_names' not in st.session_state:
    st.session_state.global_feature_names = []

# ========== FUNGSI BANTU ==========
def safe_last(series: pd.Series, default=0.0):
    return series.iloc[-1] if len(series) > 0 else default

def has_volume(volume: pd.Series) -> bool:
    return volume is not None and volume.sum() > 0

def safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^') or ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

def async_load_data(tasks: List) -> List:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(task) for task in tasks]
        results = [f.result() for f in futures]
    return results

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

# ========== LOAD DATA DENGAN CACHE ==========
@st.cache_data(ttl=CACHE_TTL, max_entries=20)
def load_data(ticker: str, timeframe: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval=timeframe, progress=False, auto_adjust=False)
        if df.empty:
            df = yf.download(ticker, period="3mo", interval=timeframe, progress=False, auto_adjust=False)
        if df.empty or len(df) < 2:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > MAX_CHART_POINTS * 2:
            df = df.tail(MAX_CHART_POINTS * 2)
        df = df.dropna(how='all')
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

# ========== SCANNER ==========
@st.cache_data(ttl=SCANNER_CACHE_TTL)
def scan_market_fast(tickers: List[str]) -> pd.DataFrame:
    try:
        all_data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True)
        rows = []
        for ticker in tickers:
            try:
                if ticker not in all_data.columns.get_level_values(0):
                    continue
                df = all_data[ticker].copy()
                df = df.dropna()
                if len(df) < 20:
                    continue
                close = df['Close']
                rsi = ta.momentum.rsi(close, window=14).iloc[-1] if len(close) >= 14 else 50
                sma20 = close.rolling(20).mean().iloc[-1]
                score_val = (1 if rsi < 35 else 0) + (1 if close.iloc[-1] > sma20 else 0)
                rows.append([ticker, round(rsi, 2), score_val])
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["Ticker", "RSI", "Score"])
    except Exception as e:
        st.error(f"Scanner error: {e}")
        return pd.DataFrame()

# ========== INDIKATOR ==========
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df['Close']
    if 'Volume' in df.columns and has_volume(df['Volume']):
        volume = df['Volume']
    else:
        volume = pd.Series(0, index=df.index)
        df['Volume'] = 0
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
        df['MACD'] = np.nan
        df['MACD_signal'] = np.nan
    if has_volume(volume):
        df['Volume_MA'] = volume.rolling(20, min_periods=1).mean()
    else:
        df['Volume_MA'] = 0.0
    df['support'] = df['Low'].rolling(20, min_periods=1).min()
    df['resistance'] = df['High'].rolling(20, min_periods=1).max()
    try:
        if has_volume(volume):
            df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'], fillna=True)
        else:
            df['AD'] = 0.0
    except Exception:
        df['AD'] = 0.0
    try:
        if has_volume(volume):
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20, fillna=True)
        else:
            df['CMF'] = 0.0
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

# ========== RULE-BASED AI SCORE (RAW CONDITION COUNT) ==========
def calculate_rule_score_raw(df: pd.DataFrame, volume: pd.Series) -> float:
    """Menghitung rule score dalam range 0-1 (berdasarkan jumlah kondisi terpenuhi)."""
    if df.empty:
        return 0
    rsi = safe_last(df['RSI'], 50)
    close = safe_last(df['Close'])
    sma20 = safe_last(df['SMA20'], close)
    sma50 = safe_last(df['SMA50'], close)
    macd = safe_last(df['MACD'], 0)
    macd_signal = safe_last(df['MACD_signal'], 0)
    vol = safe_last(volume, 0)
    vol_ma = safe_last(df['Volume_MA'], 1)
    conditions = [
        rsi < 35,
        close > sma20,
        sma20 > sma50,
        (macd > macd_signal) if not np.isnan(macd) and not np.isnan(macd_signal) else False,
        (vol > vol_ma) if has_volume(volume) else False
    ]
    return sum(conditions) / len(conditions)

# ========== MACHINE LEARNING (SINGLE TICKER) UNTUK ML SCORE ==========
def build_ml_features(df: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    features = pd.DataFrame(index=df.index)
    features['rsi'] = df['RSI']
    features['macd'] = df['MACD']
    features['macd_signal'] = df['MACD_signal']
    features['sma20'] = df['SMA20']
    features['sma50'] = df['SMA50']
    features['price'] = df['Close']
    features['volume'] = volume
    features['vol_ma'] = df['Volume_MA']
    features['return_5d'] = df['Close'].pct_change(5)
    features['volatility'] = df['Close'].pct_change().rolling(10).std()
    return features.fillna(0)

def create_labels(df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    future_return = df['Close'].shift(-forward_days) / df['Close'] - 1
    return (future_return > 0).astype(int)

def train_ml_model(features: pd.DataFrame, labels: pd.Series):
    if not SKLEARN_AVAILABLE or len(features) < 50:
        return None
    model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    model.fit(features, labels)
    return model

def ml_prediction_score(df: pd.DataFrame, volume: pd.Series, forward_days: int = 5) -> float:
    """Return ML probability in range 0-1."""
    if not SKLEARN_AVAILABLE or df.empty or len(df) < 60:
        return 0.5
    features = build_ml_features(df, volume)
    labels = create_labels(df, forward_days)
    if len(features) < 50:
        return 0.5
    train_features = features.iloc[:-forward_days]
    train_labels = labels.iloc[:-forward_days]
    if len(train_features) < 30:
        return 0.5
    model = train_ml_model(train_features, train_labels)
    if model is None:
        return 0.5
    latest = features.iloc[-1:].fillna(0)
    prob = model.predict_proba(latest)[0][1]  # probability of up
    return prob

# ========== MULTI-TICKER GLOBAL ML (UNTUK ENSEMBLE) ==========
@st.cache_data(ttl=3600)
def build_multi_ticker_dataset(tickers: List[str], period: str = "2y"):
    all_dfs = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 50:
                continue
            df = df[['Open','High','Low','Close','Volume']].copy()
            df = df.dropna()
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['return_5d'] = df['Close'].pct_change(5)
            df['volatility'] = df['Close'].pct_change().rolling(10).std()
            df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            df = df.dropna()
            all_dfs.append(df)
        except Exception:
            continue
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def train_global_model(tickers: List[str]):
    if not SKLEARN_AVAILABLE:
        return None, []
    data = build_multi_ticker_dataset(tickers)
    if data.empty:
        return None, []
    feature_cols = ['RSI', 'MACD', 'MACD_signal', 'SMA20', 'SMA50', 'Volume', 'Volume_MA', 'return_5d', 'volatility']
    feature_cols = [c for c in feature_cols if c in data.columns]
    X = data[feature_cols].fillna(0)
    y = data['target']
    if len(X) < 100:
        return None, []
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, feature_cols

def get_global_ml_probability(ticker_df: pd.DataFrame, volume: pd.Series, global_model, feature_names) -> float:
    """Return probability from global model (0-1)."""
    if global_model is None or not feature_names:
        return 0.5
    features = pd.DataFrame(index=ticker_df.index)
    features['RSI'] = ticker_df['RSI']
    features['MACD'] = ticker_df['MACD']
    features['MACD_signal'] = ticker_df['MACD_signal']
    features['SMA20'] = ticker_df['SMA20']
    features['SMA50'] = ticker_df['SMA50']
    features['Volume'] = volume
    features['Volume_MA'] = ticker_df['Volume_MA']
    features['return_5d'] = ticker_df['Close'].pct_change(5)
    features['volatility'] = ticker_df['Close'].pct_change().rolling(10).std()
    features = features.fillna(0)
    features = features[feature_names] if all(c in features.columns for c in feature_names) else pd.DataFrame()
    if features.empty:
        return 0.5
    latest = features.iloc[-1:].fillna(0)
    prob = global_model.predict_proba(latest)[0][1]
    return prob

# ========== SMART MONEY SCORE (NORMALIZED 0-1) ==========
def calculate_smart_money_score_normalized(df):
    if df.empty or len(df) < 20:
        return 0.5
    score = 0
    if safe_last(df['CMF']) > 0:
        score += 1
    if len(df) >= 5 and safe_last(df['AD']) > df['AD'].iloc[-5]:
        score += 1
    if has_volume(df['Volume']) and safe_last(df['Volume']) > safe_last(df['Volume_MA']) * 1.5:
        score += 1
    if safe_last(df['Close']) > safe_last(df['SMA20']):
        score += 1
    return score / 4.0

# ========== MACRO SCORE (NORMALIZED 0-1) ==========
def get_macro_score_normalized():
    macro_status, _ = get_macro_signal()
    if macro_status == "Risk ON":
        return 1.0
    elif macro_status == "Neutral":
        return 0.5
    else:
        return 0.0

# ========== GLOBAL MACRO CACHE (ASYNC) ==========
@st.cache_data(ttl=CACHE_TTL)
def get_macro_data_async():
    def download_ihsg():
        return yf.download("^JKSE", period="5d", progress=False)['Close']
    def download_usd():
        return yf.download("USDIDR=X", period="5d", progress=False)['Close']
    def download_nasdaq():
        return yf.download("^IXIC", period="5d", progress=False)['Close']
    ihsg, usd, nasdaq = async_load_data([download_ihsg, download_usd, download_nasdaq])
    return ihsg, usd, nasdaq

def get_macro_signal():
    try:
        ihsg, usd, nasdaq = get_macro_data_async()
        score = 0
        if safe_last(ihsg) > ihsg.mean():
            score += 1
        if safe_last(usd) < usd.mean():
            score += 1
        if safe_last(nasdaq) > nasdaq.mean():
            score += 1
        if score >= 2:
            return "Risk ON", score
        elif score == 1:
            return "Neutral", score
        else:
            return "Risk OFF", score
    except:
        return "Neutral", 1

@st.cache_data(ttl=CACHE_TTL)
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
                ret = (safe_last(close) - close.iloc[0]) / close.iloc[0]
                returns.append(float(ret))
            except Exception:
                continue
        if returns:
            sector_perf[sector] = sum(returns) / len(returns)
    if not sector_perf:
        return "Neutral", sector_perf
    best = max(sector_perf, key=sector_perf.get)
    return best, sector_perf

@st.cache_data(ttl=CACHE_TTL)
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
    df['sma'] = df['sma'].fillna(df['Close'])
    df['rsi'] = df['rsi'].fillna(50)
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
    bh_return = (safe_last(df['Close']) / df['Close'].iloc[0] - 1) * 100
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'num_trades': len(trades),
        'trades': trades,
        'equity_curve': equity_curve,
        'bh_return': bh_return
    }

def get_news_sentiment():
    if not (ENABLE_NEWS and FEEDPARSER_AVAILABLE and ENABLE_SENTIMENT and TEXTBLOB_AVAILABLE):
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
        if df.empty or len(df) < 20:
            return "Neutral"
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        if safe_last(close) > safe_last(sma20):
            return "Bullish"
        elif safe_last(close) < safe_last(sma20):
            return "Bearish"
        return "Neutral"
    return {
        "Daily": trend(daily),
        "Weekly": trend(weekly),
        "Monthly": trend(monthly)
    }

def calculate_smart_money(df):
    """Legacy function untuk smart money score 0-4 dan status."""
    if df.empty or len(df) < 20:
        return 0, "Neutral"
    score = 0
    if safe_last(df['CMF']) > 0:
        score += 1
    if len(df) >= 5 and safe_last(df['AD']) > df['AD'].iloc[-5]:
        score += 1
    if has_volume(df['Volume']) and safe_last(df['Volume']) > safe_last(df['Volume_MA']) * 1.5:
        score += 1
    if safe_last(df['Close']) > safe_last(df['SMA20']):
        score += 1
    if score >= 3:
        status = "Accumulation"
    elif score == 2:
        status = "Neutral"
    else:
        status = "Distribution"
    return score, status

def get_all_signals(df, volume, ticker):
    ai_score = calculate_ai_score(df, volume)  # rule score 0-5
    smart_score, smart_status = calculate_smart_money(df)
    macro_status, macro_score = get_macro_signal()
    best_sector, _ = get_sector_rotation()
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
    tech_score = signals['ai_score'] / 5
    smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0}
    smart_score = smart_map.get(signals['smart_status'], 0.5)
    macro_map = {"Risk ON": 1, "Neutral": 0.5, "Risk OFF": 0}
    macro_score = macro_map.get(signals['macro_status'], 0.5)
    sector_score = 1 if ticker.replace(".JK","") in signals['best_sector'] else 0.5
    returns = df['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    if vol < 0.15:
        risk_score = 1
    elif vol < 0.30:
        risk_score = 0.6
    else:
        risk_score = 0.2
    momentum = df['Close'].pct_change(5).iloc[-1] if len(df) >= 5 else 0
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

# ========== NEW ENSEMBLE AI SCORE (STEP 3) ==========
def ensemble_ai_score(df: pd.DataFrame, volume: pd.Series, ticker: str) -> Tuple[float, Dict]:
    """
    Menghitung final ensemble score (0-100) berdasarkan:
    - Rule engine (30%)
    - ML model (30%) - prioritas global model jika ada, else single ticker
    - Smart money (20%)
    - Macro (20%)
    """
    # 1. Rule score (0-1)
    rule_score = calculate_rule_score_raw(df, volume)
    
    # 2. ML score (0-1)
    # Coba gunakan global model terlebih dahulu jika tersedia
    if st.session_state.global_ml_model is not None:
        ml_prob = get_global_ml_probability(df, volume, st.session_state.global_ml_model, st.session_state.global_feature_names)
    else:
        ml_prob = ml_prediction_score(df, volume, forward_days=5)
    
    # 3. Smart money score (0-1)
    smart_score = calculate_smart_money_score_normalized(df)
    
    # 4. Macro score (0-1)
    macro_score = get_macro_score_normalized()
    
    # Bobot
    w_rule, w_ml, w_smart, w_macro = 0.30, 0.30, 0.20, 0.20
    final = (w_rule * rule_score + w_ml * ml_prob + w_smart * smart_score + w_macro * macro_score) * 100
    final = np.clip(final, 0, 100)
    
    breakdown = {
        "rule": rule_score * 100,
        "ml": ml_prob * 100,
        "smart": smart_score * 100,
        "macro": macro_score * 100
    }
    return final, breakdown

# ========== PORTFOLIO OPTIMIZER FUNCTIONS ==========
def get_portfolio_returns(tickers, period="1y"):
    data = yf.download(tickers, period=period, interval="1d", progress=False, auto_adjust=False)['Close']
    if data.empty:
        return pd.DataFrame()
    returns = data.pct_change().dropna()
    return returns

def portfolio_statistics(weights, returns, cov_matrix):
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def negative_sharpe(weights, returns, cov_matrix):
    _, _, sharpe = portfolio_statistics(weights, returns, cov_matrix)
    return -sharpe

def optimize_portfolio(returns):
    n_assets = len(returns.columns)
    init_weights = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(negative_sharpe, init_weights, args=(returns, returns.cov()),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else init_weights

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
        st.session_state.global_ml_model = None
        st.rerun()
    st.markdown("---")
    st.caption("Data dari Yahoo Finance | Update 5 menit")

# ========== TRAINING GLOBAL MODEL ==========
if ENABLE_MULTI_TICKER_ML and SKLEARN_AVAILABLE and st.session_state.global_ml_model is None:
    with st.spinner("Melatih global AI model dari seluruh IHSG (sekali saja)..."):
        model, features = train_global_model(IHSG_BLUE_CHIPS)
        if model is not None:
            st.session_state.global_ml_model = model
            st.session_state.global_feature_names = features
            st.success("Global ML model siap!")
        else:
            st.warning("Global ML model gagal dilatih. Ensemble tetap berjalan dengan model per ticker.")

# ========== LOAD DATA ==========
with st.spinner("Memuat data..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 5:
    st.warning(f"Data tidak cukup untuk {ticker} (minimal 5 periode).")
    st.stop()

data = add_indicators(data)

# Alert Check
if not data.empty:
    current_resistance = safe_last(data['resistance'])
    current_price = safe_last(data['Close'])
    if should_notify_breakout(current_price, current_resistance):
        show_notification(f"🚀 BREAKOUT! Harga menembus resistance {current_resistance:.2f}", "toast", "🚀")
        st.session_state.last_resistance = current_resistance
        st.session_state.last_breakout_notify_time = datetime.now()
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
    if has_volume(volume):
        vol_last = safe_last(volume)
        vol_ma = safe_last(data['Volume_MA'])
        if vol_ma > 0:
            volume_ratio = vol_last / vol_ma
            if should_notify_volume_spike(volume_ratio):
                show_notification(f"🔥 Volume Spike! {volume_ratio:.1f}x", "toast", "⚠️")
                st.session_state.last_volume_ratio = volume_ratio
                st.session_state.last_volume_notify_time = datetime.now()

# ========== HEADER HARGA ==========
st.title(f"📈 {ticker}")
last_close = safe_last(data['Close'])
last_high = safe_last(data['High'])
last_low = safe_last(data['Low'])
prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Terakhir", f"{last_close:.2f}", f"{change_pct:.2f}%", delta_color="normal")
col2.metric("Hari Ini - Tertinggi", f"{last_high:.2f}")
col3.metric("Hari Ini - Terendah", f"{last_low:.2f}")
col4.metric("Volume Terakhir", f"{safe_last(volume):,.0f}" if has_volume(volume) else "N/A")

ad_val = safe_last(data['AD'])
cmf_val = safe_last(data['CMF'])
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
    "🧪 Backtest", "📖 Info", "🏦 Smart Money", "🌍 Macro Market", "🔄 Sector Flow"
])

# ========== TAB 1: GRAFIK ==========
with tab1:
    st.subheader("Candlestick Chart dengan Volume")
    chart_data = data.tail(MAX_CHART_POINTS) if len(data) > MAX_CHART_POINTS else data
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
                                 low=chart_data['Low'], close=chart_data['Close'], name="Price",
                                 showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA20'], name="SMA20", line=dict(width=1), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA50'], name="SMA50", line=dict(width=1), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['support'], name="Support", line=dict(dash='dash', width=1), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['resistance'], name="Resistance", line=dict(dash='dash', width=1), showlegend=True), row=1, col=1)
    vol_data = chart_data['Volume'] if 'Volume' in chart_data.columns else pd.Series(0, index=chart_data.index)
    fig.add_trace(go.Bar(x=chart_data.index, y=vol_data, name="Volume", marker_color='lightblue', showlegend=True), row=2, col=1)
    fig.update_layout(height=600, title_text=f"{ticker} - Candlestick & Volume", showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Tanggal", row=2, col=1)
    fig.update_yaxes(title_text="Harga", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.subheader("RSI (14) & MACD")
    col_r, col_m = st.columns(2)
    with col_r:
        st.line_chart(data['RSI'].tail(100))
        st.caption("RSI < 35: Oversold | >70: Overbought")
    with col_m:
        st.line_chart(data[['MACD', 'MACD_signal']].tail(100))
        st.caption("MACD > Signal: Bullish")
    st.subheader("AD & CMF")
    col_ad, col_cmf = st.columns(2)
    with col_ad:
        st.line_chart(data['AD'].tail(100))
    with col_cmf:
        st.line_chart(data['CMF'].tail(100))

# ========== TAB 2: AI SIGNAL (DENGAN ENSEMBLE BRAIN) ==========
with tab2:
    # Hitung ensemble score
    final_score, breakdown = ensemble_ai_score(data, volume, ticker)
    
    st.subheader("🧠 ENSEMBLE TRADING BRAIN")
    st.metric("Final AI Score", f"{final_score:.1f}%")
    if final_score >= 75:
        st.success("🚀 STRONG BUY SIGNAL")
        signal = "STRONG BUY"
    elif final_score >= 60:
        st.info("🟡 BUY / HOLD")
        signal = "BUY / HOLD"
    elif final_score >= 45:
        st.warning("⚠️ NEUTRAL / WAIT")
        signal = "NEUTRAL"
    else:
        st.error("🔻 SELL / AVOID")
        signal = "SELL / AVOID"
    
    st.caption("Ensemble menggabungkan Rule Engine (30%), ML Model (30%), Smart Money (20%), Macro (20%)")
    st.divider()
    
    # Breakdown
    st.subheader("🧩 Ensemble Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rule Engine", f"{breakdown['rule']:.1f}%")
    col2.metric("ML Model", f"{breakdown['ml']:.1f}%")
    col3.metric("Smart Money", f"{breakdown['smart']:.1f}%")
    macro_status, _ = get_macro_signal()
    col4.metric("Macro", macro_status, delta=f"{breakdown['macro']:.1f}%")
    
    # Detail
    with st.expander("Lihat Detail Komponen"):
        st.write("**Rule Engine:** RSI < 35, Close > SMA20, SMA20 > SMA50, MACD > Signal, Volume > MA")
        st.write("**ML Model:** RandomForest (probabilitas naik 5 hari ke depan)")
        st.write("**Smart Money:** CMF, AD trend, volume spike, price vs SMA20")
        st.write("**Macro:** IHSG trend, USD/IDR, Nasdaq")
    
    st.divider()
    
    # Risk Meter
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

    st.subheader("📊 Probability Engine (Rule-based)")
    bull = bear = 0
    rsi = safe_last(data['RSI'], 50)
    close = safe_last(data['Close'])
    sma20 = safe_last(data['SMA20'], close)
    macd = safe_last(data['MACD'], 0)
    macd_signal = safe_last(data['MACD_signal'], 0)
    if rsi < 35:
        bull += 1
    else:
        bear += 1
    if close > sma20:
        bull += 1
    else:
        bear += 1
    if macd > macd_signal:
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
        if len(data) > 1 and safe_last(data['Close']) > data['resistance'].iloc[-2]:
            st.success("🔥 BREAKOUT DETECTED")
        else:
            st.info("Tidak ada breakout")
    with col_vol:
        st.subheader("🚨 Volume Alert")
        if has_volume(volume):
            vol_last = safe_last(volume)
            vol_ma = safe_last(data['Volume_MA'])
            if vol_last > vol_ma * st.session_state.user_volume_spike_threshold:
                st.success("Volume Spike Detected")
            else:
                st.write("Normal Volume")
        else:
            st.info("Volume tidak tersedia")

    st.header("🚀 GOD MODE TRADING ENGINE")
    price = safe_last(data['Close'])
    support = safe_last(data['support'], price * 0.95)
    resistance = safe_last(data['resistance'], price * 1.05)
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
        if price > safe_last(data['SMA20']):
            trend_score += 1
        if safe_last(data['SMA20']) > safe_last(data['SMA50']):
            trend_score += 1
        if safe_last(data['RSI'], 50) > 50:
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
    if safe_last(data['RSI'], 50) < 70:
        god_score += 1
    if price > safe_last(data['SMA20']):
        god_score += 1
    if god_score >= 3:
        st.success("🚀 GOD MODE BUY")
    elif god_score == 2:
        st.warning("⚡ SPEC BUY")
    else:
        st.error("❌ NO TRADE")

    st.header("FINAL DECISION (Ensemble)")
    if final_score >= 75:
        st.success("🟢 ACCUMULATE")
    elif final_score >= 60:
        st.warning("🟡 WAIT / HOLD")
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
    signals = get_all_signals(data, volume, ticker)
    smart_score, smart_status = signals['smart_score'], signals['smart_status']
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
    macro_status, macro_score = signals['macro_status'], signals['macro_score']
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
    best_sector = signals['best_sector']
    st.metric("Leading Sector", best_sector)
    _, sector_data = get_sector_rotation()
    for sector, val in sector_data.items():
        st.write(f"{sector}: {val:.2%}")

    st.subheader("🎯 Final AI Confidence (Legacy)")
    confidence = signals['confidence']
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
    if st.button("🔍 Scan Sekarang", use_container_width=True, type="primary"):
        with st.spinner("Memindai 15 saham..."):
            scan_df = scan_market_fast(IHSG_BLUE_CHIPS)
            st.session_state.scan_result = scan_df
    if 'scan_result' in st.session_state and not st.session_state.scan_result.empty:
        scan_df = st.session_state.scan_result
        st.dataframe(scan_df.sort_values("Score", ascending=False), use_container_width=True)
        st.subheader("🚀 TOP 5 BUY")
        top5 = scan_df.sort_values("Score", ascending=False).head(5)
        if not top5.empty:
            st.table(top5)
        else:
            st.info("Tidak ada saham dengan score > 0")
    else:
        st.info("Klik tombol 'Scan Sekarang' untuk memindai saham")

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
        portfolio_df['pnl_pct'] = np.where(
            portfolio_df['entry_price'] > 0,
            ((portfolio_df['current_price'] - portfolio_df['entry_price']) / portfolio_df['entry_price']) * 100,
            0
        )
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
    
    # Portfolio Optimizer
    st.divider()
    st.subheader("📊 Portfolio Optimizer (Markowitz)")
    st.markdown("Optimasi alokasi portofolio berdasarkan **mean-variance optimization** (Sharpe ratio maksimum).")
    if not SCIPY_AVAILABLE:
        st.warning("Library 'scipy' tidak tersedia. Optimizer tidak bisa berjalan. Silakan install scipy.")
    else:
        optimizer_mode = st.radio("Pilih aset untuk optimasi:", 
                                   ("Gunakan saham dari portfolio", "Pilih saham sendiri"),
                                   horizontal=True)
        if optimizer_mode == "Gunakan saham dari portfolio":
            if st.session_state.portfolio:
                tickers_opt = list(set([p['ticker'] for p in st.session_state.portfolio]))
                st.info(f"Optimasi untuk {len(tickers_opt)} saham: {', '.join(tickers_opt)}")
            else:
                st.warning("Portfolio kosong. Silakan tambah posisi atau pilih 'Pilih saham sendiri'.")
                tickers_opt = []
        else:
            default_tickers = IHSG_BLUE_CHIPS[:5]
            tickers_input = st.text_input("Masukkan ticker (pisahkan koma)", 
                                          value=",".join(default_tickers),
                                          help="Contoh: BBCA.JK,BBRI.JK,TLKM.JK")
            tickers_opt = [fix_ticker(t.strip()) for t in tickers_input.split(",") if t.strip()]
        if tickers_opt and len(tickers_opt) >= 2:
            period_opt = st.selectbox("Periode data untuk optimasi", ["1y", "2y", "6mo"], index=0)
            with st.spinner("Menghitung optimasi portofolio..."):
                returns_opt = get_portfolio_returns(tickers_opt, period=period_opt)
                if returns_opt.empty or len(returns_opt.columns) < 2:
                    st.error("Data tidak cukup untuk optimasi. Coba periode lain atau periksa ticker.")
                else:
                    opt_weights = optimize_portfolio(returns_opt)
                    opt_return, opt_vol, opt_sharpe = portfolio_statistics(opt_weights, returns_opt, returns_opt.cov())
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Annual Return", f"{opt_return*100:.2f}%")
                    col2.metric("Expected Volatility", f"{opt_vol*100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
                    weights_df = pd.DataFrame({
                        "Saham": tickers_opt,
                        "Bobot Optimal": opt_weights,
                        "Bobot (%)": [f"{w*100:.1f}%" for w in opt_weights]
                    }).sort_values("Bobot Optimal", ascending=False)
                    st.dataframe(weights_df, use_container_width=True)
                    fig_pie = go.Figure(data=[go.Pie(labels=tickers_opt, values=opt_weights, hole=0.4)])
                    fig_pie.update_layout(title="Alokasi Optimal", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption("Optimasi menggunakan metode Mean-Variance (Markowitz) dengan maksimasi Sharpe ratio. Asumsi risk-free rate = 0. Hasil hanya untuk edukasi.")
        else:
            st.info("Pilih minimal 2 saham untuk optimasi.")

# ========== TAB 5: BACKTEST ==========
with tab5:
    st.header("🧪 Backtesting Strategi")
    if not has_volume(volume):
        st.info("Catatan: Data volume tidak tersedia untuk indeks")
    st.subheader("📈 Backtest Strategi (RSI + SMA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_buy = st.slider("RSI Buy Threshold", min_value=20, max_value=45, value=35, step=1)
    with col2:
        rsi_sell = st.slider("RSI Sell Threshold", min_value=60, max_value=85, value=70, step=1)
    with col3:
        sma_period = st.slider("SMA Period", min_value=10, max_value=50, value=20, step=5)
    commission = st.checkbox("Include Commission (0.1% per trade)", value=False)
    commission_rate = 0.001 if commission else 0
    with st.spinner("Menjalankan backtest..."):
        bt_result = backtest_strategy(data, rsi_buy=rsi_buy, rsi_sell=rsi_sell, sma_period=sma_period, commission=commission_rate)
    if bt_result:
        col_ret, col_trades, col_bh = st.columns(3)
        col_ret.metric("Total Return", f"{bt_result['total_return']:.2f}%")
        col_trades.metric("Jumlah Transaksi", bt_result['num_trades'])
        col_bh.metric("Buy & Hold Return", f"{bt_result['bh_return']:.2f}%")
        st.write(f"**Modal Akhir:** Rp {bt_result['final_capital']:,.2f}")
        if not IS_CLOUD and len(bt_result['equity_curve']) > 0:
            st.subheader("📈 Equity Curve")
            equity_df = pd.DataFrame({'Date': data.index[:len(bt_result['equity_curve'])], 'Equity': bt_result['equity_curve']})
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Equity'], mode='lines', name='Equity'))
            fig_eq.update_layout(title="Pertumbuhan Modal", xaxis_title="Tanggal", yaxis_title="Nilai (Rp)", height=400)
            st.plotly_chart(fig_eq, use_container_width=True, config={'displayModeBar': False})
        with st.expander("Lihat Detail Transaksi"):
            if bt_result['trades']:
                st.dataframe(pd.DataFrame(bt_result['trades'], columns=['Tipe', 'Tanggal', 'Harga']))
            else:
                st.info("Tidak ada transaksi.")
    else:
        st.warning(f"Data tidak cukup untuk backtest. Minimal 50 baris, saat ini {len(data)} baris.")
    if IS_CLOUD:
        st.info("💡 **Mode Cloud:** ML global nonaktif, ensemble tetap jalan dengan model per ticker.")

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
    with st.expander("📖 Glossary (Klik untuk lihat)"):
        st.markdown("""
        **RSI** < 35: Oversold | >70: Overbought  
        **SMA20/50**: Harga > SMA = uptrend  
        **MACD > Signal**: Bullish  
        **Volume > Volume MA**: Volume di atas rata-rata  
        **AD > 0**: Akumulasi (tekanan beli)  
        **CMF > 0**: Tekanan beli  
        **Risk Reward Ratio**: Target/risk > 2 = good setup  
        **Ensemble AI Score**: Gabungan Rule (30%), ML (30%), Smart Money (20%), Macro (20%)  
        **Rule Engine**: 5 kondisi teknikal klasik  
        **ML Model**: RandomForest yang dilatih dari data IHSG (global atau per ticker)  
        **Smart Money**: Deteksi akumulasi/distribusi institusi  
        **Macro Filter**: Risk ON/OFF berdasarkan IHSG, USD/IDR, Nasdaq  
        **Multi Timeframe**: Bullish jika harga > SMA20 di daily, weekly, monthly  
        **Portfolio Optimizer**: Mean-variance optimization untuk alokasi aset optimal
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
        cmf_ok = safe_last(data['CMF']) > 0
        ad_ok = safe_last(data['AD']) > data['AD'].iloc[-5] if len(data) >= 5 else False
        vol_ok = has_volume(data['Volume']) and safe_last(data['Volume']) > safe_last(data['Volume_MA']) * 1.5
        price_ok = safe_last(data['Close']) > safe_last(data['SMA20'])
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
        ihsg, usd, nasdaq = get_macro_data_async()
        st.subheader("Data Terkini")
        col1, col2, col3 = st.columns(3)
        col1.metric("IHSG", f"{safe_last(ihsg):.2f}", f"{((safe_last(ihsg)/ihsg.iloc[0])-1)*100:.2f}%")
        col2.metric("USD/IDR", f"{safe_last(usd):.2f}", f"{((safe_last(usd)/usd.iloc[0])-1)*100:.2f}%")
        col3.metric("Nasdaq", f"{safe_last(nasdaq):.2f}", f"{((safe_last(nasdaq)/nasdaq.iloc[0])-1)*100:.2f}%")
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

gc.collect()
