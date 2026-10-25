# ========== COPY MULAI DARI SINI ==========
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
import joblib
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
ENABLE_MULTI_TICKER_ML = True 

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
    st.info("☁️ **Mode Cloud** - Performa optimal. Model ML menggunakan data per ticker.")
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
if 'scan_result' not in st.session_state:
    st.session_state.scan_result = pd.DataFrame()

# ========== FUNGSI BANTU ==========
def safe_last(series: pd.Series, default=0.0):
    if series is None or len(series) == 0:
        return default
    return series.iloc[-1]

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

@st.cache_data(ttl=3600, max_entries=20)
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
@st.cache_data(ttl=CACHE_TTL, max_entries=8)
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
@st.cache_data(ttl=SCANNER_CACHE_TTL, max_entries=5)
def scan_market_fast(tickers: List[str]) -> pd.DataFrame:
    try:
        all_data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True)
        rows = []
        for ticker in tickers:
            try:
                if isinstance(all_data.columns, pd.MultiIndex):
                    if ticker not in all_data.columns.get_level_values(0):
                        continue
                    df = all_data[ticker].copy()
                else:
                    df = all_data.copy()
                
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
    
    cols_to_fill = ['RSI', 'MACD', 'MACD_signal', 'SMA20', 'SMA50', 'Volume_MA', 'support', 'resistance', 'AD', 'CMF']
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    
    return df

# ========== RULE-BASED RAW SCORE (0-1) ==========
def calculate_rule_score_raw(df: pd.DataFrame, volume: pd.Series) -> float:
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

def calculate_ai_score(df: pd.DataFrame, volume: pd.Series) -> int:
    raw = calculate_rule_score_raw(df, volume)
    return int(round(raw * 5))

# ========== MOMENTUM REWARD SIGNAL (0-1) ==========
def momentum_reward_score(df: pd.DataFrame, lookback: int = 20) -> float:
    if df.empty or len(df) < lookback:
        return 0.5
    returns = df['Close'].pct_change().dropna()
    if len(returns) < lookback:
        return 0.5
    recent_returns = returns.iloc[-lookback:]
    reward = (recent_returns > 0).sum() / len(recent_returns)
    return float(reward)

# ========== MACHINE LEARNING (SINGLE TICKER) ==========
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
    if not SKLEARN_AVAILABLE or df.empty or len(df) < 60:
        return 0.5
    features = build_ml_features(df, volume)
    labels = create_labels(df, forward_days)
    if len(features) < 50:
        return 0.5
    
    valid = labels.notna()
    features, labels = features[valid], labels[valid].astype(int)
    
    train_features = features.iloc[:-forward_days]
    train_labels = labels.iloc[:-forward_days]
    if len(train_features) < 30:
        return 0.5
    model = train_ml_model(train_features, train_labels)
    if model is None:
        return 0.5
    latest = features.iloc[-1:].fillna(0)
    prob = model.predict_proba(latest)[0][1]
    return prob

# ========== MULTI-TICKER GLOBAL ML ==========
@st.cache_data(ttl=3600, max_entries=2)
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

# ========== SMART MONEY SCORE ==========
def calculate_smart_money_score_normalized(df):
    if df.empty or len(df) < 20:
        return 0.5
    score = 0
    max_score = 4
    if safe_last(df['CMF']) > 0:
        score += 1
    if len(df) >= 5 and safe_last(df['AD']) > df['AD'].iloc[-5]:
        score += 1
    if has_volume(df['Volume']):
        if safe_last(df['Volume']) > safe_last(df['Volume_MA']) * 1.5:
            score += 1
    else:
        max_score -= 1 
        
    if safe_last(df['Close']) > safe_last(df['SMA20']):
        score += 1
    return score / max_score if max_score > 0 else 0.5

def calculate_smart_money(df):
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

# ========== MACRO SCORE ==========
def get_macro_score_normalized():
    macro_status, _ = get_macro_signal()
    if macro_status == "Risk ON":
        return 1.0
    elif macro_status == "Neutral":
        return 0.5
    else:
        return 0.0

# ========== GLOBAL MACRO CACHE (ASYNC) ==========
@st.cache_data(ttl=CACHE_TTL, max_entries=10)
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
    except Exception as e:
        return "Neutral", 1

@st.cache_data(ttl=CACHE_TTL, max_entries=10)
def get_sector_rotation():
    sectors = {
        "Bank": ["BBRI.JK","BMRI.JK","BBCA.JK"],
        "Mining": ["ADRO.JK","ITMG.JK","ANTM.JK"],
        "Telco": ["TLKM.JK","EXCL.JK","ISAT.JK"],
        "Consumer": ["ICBP.JK","INDF.JK","UNVR.JK"]
    }
    all_tickers = [t for ts in sectors.values() for t in ts]
    
    try:
        data = yf.download(all_tickers, period="5d", group_by="ticker", progress=False, threads=True, auto_adjust=False)['Close']
    except Exception:
        return "Neutral", {}

    sector_perf = {}
    for sector, tickers in sectors.items():
        returns = []
        for t in tickers:
            try:
                if isinstance(data, pd.Series): 
                    close = data
                elif t in data.columns:
                    close = data[t]
                else:
                    continue
                    
                if len(close.dropna()) < 2:
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

@st.cache_data(ttl=CACHE_TTL, max_entries=10)
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

@st.cache_data(ttl=CACHE_TTL, max_entries=10)
def get_multi_timeframe_trend(ticker):
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

def get_all_signals(df, volume, ticker):
    ai_score = calculate_ai_score(df, volume)
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
    ticker_clean = ticker.replace(".JK","")
    if ticker_clean in sector_map.get(best_sector, []):
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
    
    sector_map = {
        "Bank": ["BBRI","BMRI","BBCA"],
        "Mining": ["ADRO","ITMG","ANTM"],
        "Telco": ["TLKM","EXCL","ISAT"],
        "Consumer": ["ICBP","INDF","UNVR"],
    }
    best = signals['best_sector']
    ticker_clean = ticker.replace(".JK","")
    sector_score = 1 if ticker_clean in sector_map.get(best, []) else 0.5

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

# ========== NEW ENSEMBLE AI SCORE ==========
def ensemble_ai_score(df: pd.DataFrame, volume: pd.Series, ticker: str) -> Tuple[float, Dict]:
    rule_score = calculate_rule_score_raw(df, volume)
    if st.session_state.global_ml_model is not None:
        ml_prob = get_global_ml_probability(df, volume, st.session_state.global_ml_model, st.session_state.global_feature_names)
    else:
        ml_prob = ml_prediction_score(df, volume, forward_days=5)
    smart_score = calculate_smart_money_score_normalized(df)
    macro_score = get_macro_score_normalized()
    reward_score = momentum_reward_score(df, lookback=20)
    w_rule, w_ml, w
