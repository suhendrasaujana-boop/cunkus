# ========== COPY MULAI DARI SINI ==========
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from plotly.subplots import make_subplots
import os
import gc
import joblib

# ========== TAMBAHAN UNTUK PORTFOLIO OPTIMIZER ==========
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ========== DETEKSI ENVIRONMENT ==========
IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE', '').lower() == 'sharing'

# ========== FITUR YANG DINONAKTIFKAN DI CLOUD ==========
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

# ========== PARAMETER BREAKOUT (BARU) ==========
BREAKOUT_ATR_MULT = 0.5   # minimal 0.5×ATR di atas resistance
BREAKOUT_MIN_PCT = 0.005  # minimal 0.5% di atas resistance
ATR_STOP_MULT = 2.0       # Stop Loss = Entry − 2×ATR
MAX_RISK_PCT_WARNING = 2.0  # batas toleransi risiko per trade

SECTOR_MAP = {
    "Bank": ["BBRI", "BMRI", "BBCA"],
    "Mining": ["ADRO", "ITMG", "ANTM", "MDKA"],
    "Telco": ["TLKM", "EXCL", "ISAT"],
    "Consumer": ["ICBP", "INDF", "UNVR", "CPIN", "JPFA"],
    "Auto": ["ASII"],
    "Cement": ["SMGR"],
    "Energy": ["UNTR"],
}

st.set_page_config(
    layout="wide",
    page_title="Smart Market Dashboard",
    page_icon="📊",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# ========== SESSION STATE ==========
if 'last_resistance' not in st.session_state:
    st.session_state.last_resistance = None
if 'last_breakout_notify_time' not in st.session_state:
    st.session_state.last_breakout_notify_time = None
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
if '_last_ticker' not in st.session_state:
    st.session_state._last_ticker = None

# ========== FUNGSI BANTU ==========
def safe_last(series, default=0.0):
    if series is None:
        return default
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return default
        try:
            series = series.squeeze()
        except Exception:
            return default
    if isinstance(series, pd.Series):
        if len(series) == 0:
            return default
        val = series.iloc[-1]
        if pd.isna(val):
            return default
        return val
    if isinstance(series, (int, float, np.integer, np.floating)):
        return default if pd.isna(series) else series
    return default

def safe_mean(series, default=0.0):
    if series is None:
        return default
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return default
        series = series.squeeze()
    if isinstance(series, pd.Series):
        if len(series) == 0:
            return default
        val = series.mean()
        return default if pd.isna(val) else val
    if isinstance(series, (int, float, np.integer, np.floating)):
        return default if pd.isna(series) else series
    return default

def has_volume(volume: pd.Series) -> bool:
    return volume is not None and isinstance(volume, pd.Series) and volume.sum() > 0

def safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def fix_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith('^') or ticker.endswith('.JK'):
        return ticker
    return ticker + '.JK'

# ========== BREAKOUT CONFIRMATION (BARU) ==========
def is_confirmed_breakout(df: pd.DataFrame,
                          atr_mult: float = BREAKOUT_ATR_MULT,
                          min_pct: float = BREAKOUT_MIN_PCT) -> Tuple[bool, float]:
    """
    Deteksi breakout yang TERKONFIRMASI:
    - Harga close (last bar) harus > resistance bar sebelumnya
    - Kelebihan harus melewati threshold: max(min_pct × resistance, atr_mult × ATR)
    Mengembalikan (is_breakout, excess_pct).
    """
    if df is None or df.empty or len(df) < 2:
        return False, 0.0
    last_close = safe_last(df['Close'])
    prev_resistance = df['resistance'].iloc[-2]
    if prev_resistance is None or prev_resistance <= 0 or pd.isna(prev_resistance):
        return False, 0.0
    atr = safe_last(df['ATR'], 0.0) if 'ATR' in df.columns else 0.0
    threshold = max(prev_resistance * min_pct, atr_mult * atr if atr > 0 else 0.0)
    if last_close > prev_resistance + threshold:
        excess_pct = (last_close - prev_resistance) / prev_resistance * 100
        return True, float(excess_pct)
    return False, 0.0

def breakout_status(df: pd.DataFrame) -> Dict[str, Any]:
    """Status breakout lengkap untuk ditampilkan di UI."""
    if df is None or df.empty or len(df) < 2:
        return {"state": "no_data", "excess": 0.0, "threshold": 0.0}
    last_close = safe_last(df['Close'])
    prev_resistance = df['resistance'].iloc[-2]
    if prev_resistance is None or pd.isna(prev_resistance) or prev_resistance <= 0:
        return {"state": "no_data", "excess": 0.0, "threshold": 0.0}
    atr = safe_last(df['ATR'], 0.0) if 'ATR' in df.columns else 0.0
    threshold = max(prev_resistance * BREAKOUT_MIN_PCT, BREAKOUT_ATR_MULT * atr if atr > 0 else 0.0)
    excess = last_close - prev_resistance
    if excess > threshold:
        return {"state": "confirmed", "excess": excess, "threshold": threshold,
                "excess_pct": excess / prev_resistance * 100,
                "resistance": prev_resistance, "atr": atr}
    elif excess > 0:
        return {"state": "pending", "excess": excess, "threshold": threshold,
                "excess_pct": excess / prev_resistance * 100,
                "resistance": prev_resistance, "atr": atr}
    else:
        return {"state": "none", "excess": excess, "threshold": threshold,
                "resistance": prev_resistance, "atr": atr}

def should_notify_breakout(current_price: float, resistance: float,
                            atr: float = 0.0, min_pct: float = BREAKOUT_MIN_PCT) -> bool:
    threshold = max(resistance * min_pct, BREAKOUT_ATR_MULT * atr if atr > 0 else 0.0)
    if current_price <= resistance + threshold:
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

def get_ticker_sector(ticker: str) -> Optional[str]:
    t_clean = ticker.replace(".JK", "").replace("^", "").upper()
    for sector, stocks in SECTOR_MAP.items():
        if t_clean in stocks:
            return sector
    return None

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
@st.cache_data(ttl=SCANNER_CACHE_TTL, max_entries=5)
def scan_market_fast(tickers: List[str]) -> pd.DataFrame:
    try:
        all_data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker",
                               progress=False, threads=True)
        if all_data is None or all_data.empty:
            return pd.DataFrame()
        is_multi = isinstance(all_data.columns, pd.MultiIndex)
        rows = []
        for ticker in tickers:
            try:
                if is_multi:
                    if ticker not in all_data.columns.get_level_values(0):
                        continue
                    df = all_data[ticker].copy()
                else:
                    df = all_data.copy()
                df = df.dropna()
                if len(df) < 20:
                    continue
                close = df['Close'].squeeze()
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

# ========== INDIKATOR (DITAMBAH ATR) ==========
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

    # ===== ATR (Average True Range) — BARU =====
    try:
        if len(df) >= 14 and 'High' in df.columns and 'Low' in df.columns:
            df['ATR'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close'], window=14
            )
        else:
            df['ATR'] = np.nan
    except Exception:
        df['ATR'] = np.nan

    df = df.ffill()
    if 'RSI' in df.columns:
        df['RSI'] = df['RSI'].fillna(50)
    if 'MACD' in df.columns:
        df['MACD'] = df['MACD'].fillna(0)
    if 'MACD_signal' in df.columns:
        df['MACD_signal'] = df['MACD_signal'].fillna(0)
    df = df.fillna(0)
    return df

# ========== RULE-BASED RAW SCORE (0-1) ==========
def calculate_rule_score_raw(df: pd.DataFrame, volume: pd.Series) -> float:
    if df.empty:
        return 0.0
    rsi = safe_last(df['RSI'], 50)
    close = safe_last(df['Close'])
    sma20 = safe_last(df['SMA20'], close)
    sma50 = safe_last(df['SMA50'], close)
    macd = safe_last(df['MACD'], 0)
    macd_signal = safe_last(df['MACD_signal'], 0)
    vol = safe_last(volume, 0)
    vol_ma = safe_last(df['Volume_MA'], 0)

    if rsi < 30 and close < sma20:
        return 0.0

    trend_conditions = [
        close > sma20,
        sma20 > sma50,
        (macd > macd_signal) if not (np.isnan(macd) or np.isnan(macd_signal)) else False,
    ]
    trend_score = sum(trend_conditions) / len(trend_conditions)
    meanrev_bonus = 1.0 if rsi < 40 else 0.0
    vol_bonus = 1.0 if (has_volume(volume) and vol_ma > 0 and vol > vol_ma) else 0.0

    return 0.65 * trend_score + 0.20 * meanrev_bonus + 0.15 * vol_bonus

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
    freq = (recent_returns > 0).sum() / len(recent_returns)
    std = recent_returns.std()
    magnitude = (recent_returns.mean() / std) if std > 0 else 0.0
    magnitude_norm = float(np.clip(magnitude + 0.5, 0.0, 1.0))
    return float(0.5 * freq + 0.5 * magnitude_norm)

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
    label = (future_return > 0).astype(int)
    label[future_return.isna()] = -1
    return label

def train_ml_model(features: pd.DataFrame, labels: pd.Series):
    if not SKLEARN_AVAILABLE or len(features) < 50:
        return None
    try:
        if labels.nunique() < 2:
            return None
        if len(features) >= 100:
            X_tr, X_val, y_tr, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            probe = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
            probe.fit(X_tr, y_tr)
            try:
                _ = accuracy_score(y_val, probe.predict(X_val))
            except Exception:
                pass
        model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
        model.fit(features, labels)
        return model
    except Exception:
        return None

def ml_prediction_score(df: pd.DataFrame, volume: pd.Series, ticker: str, forward_days: int = 5) -> float:
    if not SKLEARN_AVAILABLE or df.empty or len(df) < 60:
        return 0.5

    sig = f"{ticker}:{len(df)}:{safe_last(df['Close'], 0):.4f}"
    cache_key = f"_ml_pred_{ticker}"
    cached = st.session_state.get(cache_key)
    if cached and cached.get('sig') == sig:
        return cached['prob']

    features = build_ml_features(df, volume)
    labels = create_labels(df, forward_days)
    if len(features) < 50:
        return 0.5

    valid = labels >= 0
    X = features[valid]
    y = labels[valid]
    if len(X) < 30:
        return 0.5

    model = train_ml_model(X, y)
    if model is None:
        return 0.5

    latest = features.iloc[-1:].fillna(0)
    try:
        prob = float(model.predict_proba(latest)[0][1])
    except Exception:
        return 0.5

    st.session_state[cache_key] = {'sig': sig, 'prob': prob}
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
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
    return pd.concat(all_dfs, ignore_index=True)

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
    if len(X) < 100 or y.nunique() < 2:
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
    if not all(c in features.columns for c in feature_names):
        return 0.5
    latest = features[feature_names].iloc[-1:].fillna(0)
    try:
        prob = global_model.predict_proba(latest)[0][1]
        return float(prob)
    except Exception:
        return 0.5

# ========== SMART MONEY SCORE (DIPERBAIKI: status eksplisit data kurang) ==========
def calculate_smart_money_score_normalized(df):
    if df is None or df.empty or len(df) < 20:
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

def calculate_smart_money(df):
    # Validasi eksplisit: bukan nilai netral yang mengecoh
    if df is None or df.empty or len(df) < 20:
        return 0, "Data Tidak Cukup"
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

# ========== GLOBAL MACRO CACHE ==========
@st.cache_data(ttl=CACHE_TTL, max_entries=10)
def get_macro_data_async():
    empty = pd.Series(dtype=float)
    tickers = ["^JKSE", "USDIDR=X", "^IXIC"]
    try:
        raw = yf.download(tickers, period="5d", progress=False,
                          auto_adjust=False, group_by="ticker", threads=False)
    except Exception:
        return empty, empty, empty
    if raw is None or raw.empty:
        return empty, empty, empty

    def _extract(ticker):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    return empty
                sub = raw[ticker]
            else:
                sub = raw
            if 'Close' not in sub.columns:
                return empty
            s = sub['Close']
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.dropna()
        except Exception:
            return empty

    return _extract("^JKSE"), _extract("USDIDR=X"), _extract("^IXIC")

@st.cache_data(ttl=CACHE_TTL, max_entries=10)
def get_macro_signal():
    try:
        ihsg, usd, nasdaq = get_macro_data_async()
        score = 0
        if len(ihsg) > 0 and safe_last(ihsg) > safe_mean(ihsg):
            score += 1
        if len(usd) > 0 and safe_last(usd) < safe_mean(usd):
            score += 1
        if len(nasdaq) > 0 and safe_last(nasdaq) > safe_mean(nasdaq):
            score += 1
        if score >= 2:
            return "Risk ON", score
        elif score == 1:
            return "Neutral", score
        else:
            return "Risk OFF", score
    except Exception:
        return "Neutral", 1

@st.cache_data(ttl=CACHE_TTL, max_entries=10)
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
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close = df['Close'].squeeze()
                if isinstance(close, pd.DataFrame) or len(close) < 2:
                    continue
                first = close.iloc[0]
                if first == 0 or pd.isna(first):
                    continue
                ret = (safe_last(close) - first) / first
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
                if hist is None or hist.empty or 'Close' not in hist.columns:
                    prices[t] = None
                    continue
                close = hist['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.squeeze()
                val = safe_last(close, None)
                prices[t] = float(val) if val is not None else None
            except Exception:
                prices[t] = None
        return prices
    except Exception as e:
        st.error(f"Error fetching portfolio prices: {e}")
        return {t: None for t in tickers}

# ========== BACKTEST (DIPERLUAS: MDD + Profit Factor + Win Rate) ==========
def backtest_strategy(df, initial_capital=1000000, rsi_buy=35, rsi_sell=70,
                      sma_period=20, commission=0.001):
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
    realized_pnls = []
    pending_entry = None  # (entry_price, shares, cost)

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if df['buy_signal'].iloc[i] and position == 0:
            shares_bought = cash / (price * (1 + commission))
            pending_entry = (price, shares_bought, cash)
            position = shares_bought
            cash = 0
            trades.append(('BUY', df.index[i], price))
        elif df['sell_signal'].iloc[i] and position > 0:
            proceeds = position * price * (1 - commission)
            if pending_entry is not None:
                realized_pnls.append(proceeds - pending_entry[2])
                pending_entry = None
            cash = proceeds
            position = 0
            trades.append(('SELL', df.index[i], price))
        pos_value = position * price * (1 - commission) if position > 0 else 0
        total_value = cash + pos_value
        equity_curve.append(total_value)

    # Force close di akhir periode
    if position > 0:
        final_price = df['Close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        if pending_entry is not None:
            realized_pnls.append(proceeds - pending_entry[2])
            pending_entry = None
        cash = proceeds
        position = 0
        trades.append(('SELL', df.index[-1], final_price))
        if len(equity_curve) > 0:
            equity_curve[-1] = cash
        else:
            equity_curve.append(cash)

    final_capital = cash
    total_return = (final_capital - initial_capital) / initial_capital * 100
    first_close = df['Close'].iloc[0]
    if pd.notna(first_close) and first_close > 0:
        bh_return = (safe_last(df['Close']) / first_close - 1) * 100
    else:
        bh_return = 0

    # ===== Maximum Drawdown =====
    if equity_curve:
        eq = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq)
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns = np.where(running_max > 0, (eq - running_max) / running_max, 0.0)
        mdd = float(drawdowns.min() * 100)
    else:
        mdd = 0.0

    # ===== Profit Factor & Win Rate =====
    if realized_pnls:
        winning = sum(p for p in realized_pnls if p > 0)
        losing = abs(sum(p for p in realized_pnls if p < 0))
        if losing > 0:
            profit_factor = winning / losing
        elif winning > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0
        win_rate = sum(1 for p in realized_pnls if p > 0) / len(realized_pnls) * 100
    else:
        profit_factor = 0.0
        win_rate = 0.0

    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'num_trades': len(trades),
        'trades': trades,
        'equity_curve': equity_curve,
        'bh_return': bh_return,
        'mdd': mdd,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'realized_pnls': realized_pnls,
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
    elif smart_status == "Data Tidak Cukup":
        confidence = max(0, confidence - 10)  # penalti: data kurang
    if macro_status == "Risk ON":
        confidence += 10
    elif macro_status == "Risk OFF":
        confidence -= 10
    ticker_sector = get_ticker_sector(ticker)
    if ticker_sector is not None and ticker_sector == best_sector:
        confidence += 5
    confidence = max(0, min(80, confidence))
    confidence = int(round(confidence * 100 / 80))
    return {
        'ai_score': ai_score,
        'smart_score': smart_score,
        'smart_status': smart_status,
        'macro_status': macro_status,
        'macro_score': macro_score,
        'best_sector': best_sector,
        'confidence': confidence
    }

def weighted_decision_engine(df, volume, ticker, timeframe="1d"):
    signals = get_all_signals(df, volume, ticker)
    tech_score = signals['ai_score'] / 5
    smart_map = {"Accumulation": 1, "Neutral": 0.5, "Distribution": 0, "Data Tidak Cukup": 0.5}
    smart_score = smart_map.get(signals['smart_status'], 0.5)
    macro_map = {"Risk ON": 1, "Neutral": 0.5, "Risk OFF": 0}
    macro_score = macro_map.get(signals['macro_status'], 0.5)
    ticker_sector = get_ticker_sector(ticker)
    if ticker_sector is None:
        sector_score = 0.5
    elif ticker_sector == signals['best_sector']:
        sector_score = 1.0
    else:
        sector_score = 0.0
    returns = df['Close'].pct_change().dropna()
    pp = {"1d": 252, "1wk": 52, "1mo": 12}.get(timeframe, 252)
    vol = returns.std() * np.sqrt(pp) if len(returns) > 1 else 0
    if vol < 0.15:
        risk_score = 1
    elif vol < 0.30:
        risk_score = 0.6
    else:
        risk_score = 0.2
    momentum = df['Close'].pct_change(5).iloc[-1] if len(df) >= 5 else 0
    if pd.isna(momentum):
        momentum = 0
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

# ========== ENSEMBLE AI SCORE ==========
def ensemble_ai_score(df: pd.DataFrame, volume: pd.Series, ticker: str) -> Tuple[float, Dict]:
    rule_score = calculate_rule_score_raw(df, volume)
    if st.session_state.global_ml_model is not None:
        ml_prob = get_global_ml_probability(df, volume, st.session_state.global_ml_model, st.session_state.global_feature_names)
    else:
        ml_prob = ml_prediction_score(df, volume, ticker, forward_days=5)
    smart_score = calculate_smart_money_score_normalized(df)
    macro_score = get_macro_score_normalized()
    reward_score = momentum_reward_score(df, lookback=20)
    w_rule, w_ml, w_smart, w_macro, w_reward = 0.25, 0.25, 0.20, 0.20, 0.10
    final = (w_rule * rule_score + w_ml * ml_prob + w_smart * smart_score +
             w_macro * macro_score + w_reward * reward_score) * 100
    final = float(np.clip(final, 0, 100))
    breakdown = {
        "rule": rule_score * 100,
        "ml": ml_prob * 100,
        "smart": smart_score * 100,
        "macro": macro_score * 100,
        "reward": reward_score * 100
    }
    return final, breakdown

# ========== PORTFOLIO OPTIMIZER ==========
@st.cache_data(ttl=1800, max_entries=5)
def get_portfolio_returns(tickers_tuple: Tuple[str, ...], period: str = "1y"):
    tickers = list(tickers_tuple)
    data = yf.download(tickers, period=period, interval="1d", progress=False, auto_adjust=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if data.empty:
        return pd.DataFrame()
    return data.pct_change().dropna()

def portfolio_statistics(weights, returns, cov_matrix):
    port_return = np.sum(returns.mean() * weights) * 252
    port_var = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
    port_vol = float(np.sqrt(max(0.0, port_var)))
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
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-9})
    return result.x if result.success else init_weights

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# 📊 Smart Market Dashboard")
    ticker_input = st.text_input("Ticker", DEFAULT_TICKER, help="Contoh: BBCA, BBRI, ASII, atau ^JKSE untuk IHSG.")
    ticker = fix_ticker(ticker_input)
    if ticker != ticker_input:
        st.info(f"Format: {ticker}")

    if st.session_state._last_ticker != ticker:
        st.session_state._last_ticker = ticker
        st.session_state.last_resistance = None
        st.session_state.last_breakout_notify_time = None
        st.session_state.last_volume_notify_time = None

    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    st.markdown("### ⚙️ Pengaturan Alert")

    st.slider(
        "Volume spike multiplier", 1.0, 5.0, step=0.1,
        key="user_volume_spike_threshold",
    )
    st.number_input(
        "Breakout cooldown (jam)", 1, 72,
        key="user_breakout_cooldown_hours",
    )

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.global_ml_model = None
        for k in list(st.session_state.keys()):
            if k.startswith("_ml_pred_"):
                del st.session_state[k]
        st.rerun()

    st.markdown("---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "global_model.joblib")
    if st.session_state.global_ml_model is None:
        if os.path.exists(model_path):
            try:
                saved = joblib.load(model_path)
                st.session_state.global_ml_model = saved['model']
                st.session_state.global_feature_names = saved['feature_names']
                st.success("✅ Global ML model ready")
            except Exception as e:
                st.error(f"❌ Gagal memuat model: {e}")
        else:
            st.caption("ℹ️ Model global tidak ditemukan — pakai ML per ticker.")
    else:
        st.caption("✅ Global ML model loaded")

    st.markdown("---")
    st.caption("Data dari Yahoo Finance | Update 5 menit")

# ========== LOAD DATA ==========
with st.spinner("Memuat data..."):
    data = load_data(ticker, TIMEFRAMES[timeframe])

if data.empty or len(data) < 5:
    st.warning(f"Data tidak cukup untuk {ticker} (minimal 5 periode).")
    st.stop()

data = add_indicators(data)

# ===== VALIDASI DATA (BARU): peringatan jika data terbatas (saham baru/illikuid) =====
if len(data) < 50:
    st.warning(
        f"⚠️ **Data historis terbatas ({len(data)} baris < 50).** "
        f"Saham baru (IPO) atau illikuid mungkin menghasilkan sinyal kurang akurat. "
        f"SMA50, RSI, dan ML belum sepenuhnya andal."
    )

# ========== ALERT CHECK (DIPERBAIKI: breakout dengan konfirmasi ATR) ==========
if not data.empty and len(data) > 1:
    current_price = safe_last(data['Close'])
    prev_resistance = data['resistance'].iloc[-2]
    atr_val_alert = safe_last(data['ATR'], 0.0)
    if should_notify_breakout(current_price, prev_resistance, atr_val_alert):
        excess_pct = (current_price - prev_resistance) / prev_resistance * 100
        show_notification(
            f"🚀 CONFIRMED BREAKOUT! +{excess_pct:.2f}% di atas resistance "
            f"{prev_resistance:.2f} (filter ATR aktif)",
            "toast", "🚀"
        )
        st.session_state.last_resistance = prev_resistance
        st.session_state.last_breakout_notify_time = datetime.now()
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
    if has_volume(volume):
        vol_last = safe_last(volume)
        vol_ma = safe_last(data['Volume_MA'])
        if vol_ma > 0:
            volume_ratio = vol_last / vol_ma
            if should_notify_volume_spike(volume_ratio):
                show_notification(f"🔥 Volume Spike! {volume_ratio:.1f}x", "toast", "⚠️")
                st.session_state.last_volume_notify_time = datetime.now()
else:
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)

# ========== HEADER HARGA ==========
st.title(f"📈 {ticker}")
last_close = safe_last(data['Close'])
last_high = safe_last(data['High'])
last_low = safe_last(data['Low'])
prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
atr_display = safe_last(data['ATR'], 0.0) if 'ATR' in data.columns else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Terakhir", f"{last_close:.2f}", f"{change_pct:.2f}%", delta_color="normal")
col2.metric("Hari Ini - Tertinggi", f"{last_high:.2f}")
col3.metric("Hari Ini - Terendah", f"{last_low:.2f}")
col4.metric("ATR (14)", f"{atr_display:.2f}",
            help="Average True Range — ukuran volatilitas. Dipakai untuk filter breakout & stop loss.")

ad_val = safe_last(data['AD'])
cmf_val = safe_last(data['CMF'])
ad_status = "Akumulasi" if ad_val > 0 else "Distribusi" if ad_val < 0 else "Netral"
cmf_status = "Akumulasi" if cmf_val > 0 else "Distribusi" if cmf_val < 0 else "Netral"
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <b>📊 Akumulasi/Distribusi (AD):</b> {ad_status} ({ad_val:.2f}) &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>💰 Chaikin Money Flow (CMF20):</b> {cmf_status} ({cmf_val:.3f}) &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>📏 ATR(14):</b> {atr_display:.2f}
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
    st.subheader("AD, CMF & ATR")
    col_ad, col_cmf, col_atr = st.columns(3)
    with col_ad:
        st.line_chart(data['AD'].tail(100))
        st.caption("Akumulasi/Distribusi")
    with col_cmf:
        st.line_chart(data['CMF'].tail(100))
        st.caption("Chaikin Money Flow")
    with col_atr:
        st.line_chart(data['ATR'].tail(100))
        st.caption("ATR (14) — Volatilitas")

# ========== TAB 2: AI SIGNAL ==========
with tab2:
    # Data quality warning
    if len(data) < 50:
        st.warning(
            f"⚠️ **Data historis terbatas ({len(data)} baris < 50).** "
            f"Sinyal ML dan Smart Money mungkin kurang akurat."
        )

    final_score, breakdown = ensemble_ai_score(data, volume, ticker)
    signals = get_all_signals(data, volume, ticker)

    st.header("🎯 PRIMARY DECISION (Ensemble)")
    st.metric("Final AI Score", f"{final_score:.1f}%")
    if final_score >= 75:
        st.success("🚀 STRONG BUY SIGNAL")
    elif final_score >= 60:
        st.info("🟡 BUY / HOLD")
    elif final_score >= 45:
        st.warning("⚠️ NEUTRAL / WAIT")
    else:
        st.error("🔻 SELL / AVOID")
    st.caption("Ensemble: Rule (25%) + ML (25%) + Smart Money (20%) + Macro (20%) + Momentum Reward (10%)")

    st.subheader("🧩 Ensemble Breakdown")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rule Engine", f"{breakdown['rule']:.1f}%")
    col2.metric("ML Model", f"{breakdown['ml']:.1f}%")
    col3.metric("Smart Money", f"{breakdown['smart']:.1f}%")
    col4.metric("Macro", signals['macro_status'], delta=f"{breakdown['macro']:.1f}%")
    col5.metric("Momentum Reward", f"{breakdown['reward']:.1f}%")

    with st.expander("Lihat Detail Komponen"):
        st.write("**Rule Engine:** Trend (Close>SMA20, SMA20>SMA50, MACD>Signal) + bonus RSI oversold + konfirmasi volume. RSI<30 & harga<SMA20 → di-blacklist (falling knife).")
        st.write("**ML Model:** RandomForest (probabilitas naik 5 hari ke depan, dengan validasi train/test).")
        st.write("**Smart Money:** CMF, AD trend, volume spike, harga vs SMA20.")
        st.write("**Macro:** IHSG trend, USD/IDR, Nasdaq.")
        st.write("**Momentum Reward:** Frekuensi hari naik + magnitude (Sharpe-like) dalam 20 hari terakhir.")

    st.divider()
    st.subheader("🎯 Risk Meter")
    returns = data['Close'].pct_change().dropna()
    if len(returns) > 0:
        pp = {"1d": 252, "1wk": 52, "1mo": 12}.get(timeframe, 252)
        annual_vol = returns.std() * np.sqrt(pp) * 100
    else:
        annual_vol = 0.0
    if annual_vol < 15:
        st.success(f"Risk Level: LOW ({annual_vol:.1f}% annual)")
    elif annual_vol < 30:
        st.warning(f"Risk Level: MEDIUM ({annual_vol:.1f}% annual)")
    else:
        st.error(f"Risk Level: HIGH ({annual_vol:.1f}% annual)")

    st.subheader("📊 Probability Engine (Rule-based)")
    bull = bear = 0
    rsi = safe_last(data['RSI'], 50)
    close = safe_last(data['Close'])
    sma20 = safe_last(data['SMA20'], close)
    macd = safe_last(data['MACD'], 0)
    macd_signal = safe_last(data['MACD_signal'], 0)
    if rsi < 35: bull += 1
    else: bear += 1
    if close > sma20: bull += 1
    else: bear += 1
    if macd > macd_signal: bull += 1
    else: bear += 1
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

    # ===== BREAKOUT DETECTOR (DIPERBAIKI: konfirmasi closing + ATR) =====
    col_brk, col_vol = st.columns(2)
    with col_brk:
        st.subheader("🚀 Breakout Detector")
        st.caption("Konfirmasi: closing di atas resistance + filter ATR (0.5×ATR atau 0.5% dari resistance).")
        bs = breakout_status(data)
        if bs['state'] == "confirmed":
            st.success(
                f"✅ CONFIRMED BREAKOUT (+{bs['excess_pct']:.2f}% di atas resistance "
                f"{bs['resistance']:.2f}, threshold {bs['threshold']:.2f})"
            )
        elif bs['state'] == "pending":
            st.warning(
                f"⚠️ Breakout BELUM terkonfirmasi: excess {bs['excess']:.2f} "
                f"< threshold {bs['threshold']:.2f} (butuh ≥ {bs['excess_pct']:.2f}% atau 0.5×ATR)"
            )
        elif bs['state'] == "no_data":
            st.info("Data tidak cukup untuk deteksi breakout.")
        else:
            st.info("Tidak ada breakout (harga masih di bawah resistance).")
    with col_vol:
        st.subheader("🚨 Volume Alert")
        if has_volume(volume):
            vol_last = safe_last(volume)
            vol_ma = safe_last(data['Volume_MA'])
            if vol_ma > 0 and vol_last > vol_ma * st.session_state.user_volume_spike_threshold:
                st.success("Volume Spike Detected")
            else:
                st.write("Normal Volume")
        else:
            st.info("Volume tidak tersedia")

    st.divider()
    st.subheader("🔎 Konfirmasi Sekunder (bukan rekomendasi utama)")
    st.caption("Panel di bawah ini adalah konfirmasi tambahan. Gunakan **PRIMARY DECISION** di atas sebagai rekomendasi utama.")

    st.header("🚀 GOD MODE TRADING ENGINE (Secondary)")
    price = safe_last(data['Close'])
    support = safe_last(data['support'], price * 0.95)
    resistance = safe_last(data['resistance'], price * 1.05)
    entry = (support + price) / 2

    # ===== ATR-BASED STOP LOSS (BARU) =====
    atr_val = safe_last(data['ATR'], 0.0)
    if atr_val > 0:
        sl_atr = entry - ATR_STOP_MULT * atr_val
        if 0 < sl_atr < entry:
            stoploss = sl_atr
            sl_method = f"ATR-based (Entry − {ATR_STOP_MULT:.0f}×ATR)"
        else:
            stoploss = support * 0.97
            sl_method = "Fallback support-based (SL ATR tidak valid)"
    else:
        stoploss = support * 0.97
        sl_method = "Fallback support-based (ATR tidak tersedia)"

    target1 = resistance
    risk_amount = entry - stoploss
    reward = target1 - entry
    rr = reward / risk_amount if risk_amount > 0 else 0
    col_e, col_s, col_t = st.columns(3)
    col_e.metric("🎯 Smart Entry", f"{entry:.2f}")
    col_s.metric("🛑 Stoploss", f"{stoploss:.2f}", help=sl_method)
    col_t.metric("💰 Target 1", f"{target1:.2f}")
    st.caption(f"**Stop Loss method:** {sl_method} | ATR(14) = {atr_val:.2f}")
    st.write(f"**Risk Reward Ratio:** 1 : {rr:.2f}")
    if rr > 2:
        st.success("Good Trade Setup")
    elif rr > 1:
        st.warning("Moderate Setup")
    else:
        st.error("Bad Setup")

    col_ts, col_bs2 = st.columns(2)
    with col_ts:
        st.subheader("📈 Trend Score")
        trend_score = 0
        if price > safe_last(data['SMA20']): trend_score += 1
        if safe_last(data['SMA20']) > safe_last(data['SMA50']): trend_score += 1
        if safe_last(data['RSI'], 50) > 50: trend_score += 1
        st.write(f"Trend Score: {trend_score}/3")
    with col_bs2:
        st.subheader("🔥 Breakout Strength")
        bs2 = breakout_status(data)
        if bs2['state'] == "confirmed":
            st.success(f"✅ CONFIRMED (+{bs2['excess_pct']:.2f}%, threshold {bs2['threshold']:.2f})")
        elif bs2['state'] == "pending":
            st.warning(f"⚠️ Belum terkonfirmasi (+{bs2['excess_pct']:.2f}%, butuh ≥ {bs2['threshold']:.2f})")
        else:
            st.write("No Breakout")

    st.subheader("🧠 FINAL GOD SIGNAL")
    god_score = 0
    if rr > 2: god_score += 1
    if trend_score >= 2: god_score += 1
    if safe_last(data['RSI'], 50) < 70: god_score += 1
    if price > safe_last(data['SMA20']): god_score += 1
    if god_score >= 3:
        st.success("🚀 GOD MODE BUY")
    elif god_score == 2:
        st.warning("⚡ SPEC BUY")
    else:
        st.error("❌ NO TRADE")

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
    smart_score = signals['smart_score']
    smart_status = signals['smart_status']
    col1, col2 = st.columns(2)
    col1.metric("Smart Money Score", f"{smart_score}/4")
    col2.metric("Status", smart_status)
    if smart_status == "Accumulation":
        st.success("💰 Smart Money Accumulating")
    elif smart_status == "Distribution":
        st.error("⚠️ Smart Money Distributing")
    elif smart_status == "Data Tidak Cukup":
        st.warning("⚠️ Data historis kurang (<20 baris). Smart Money tidak dapat dihitung andal — saham baru/illikuid.")
    else:
        st.info("Sideways / Neutral")

    st.subheader("🌍 Macro Market Filter")
    macro_status = signals['macro_status']
    macro_score = signals['macro_score']
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

    st.subheader("🎯 Legacy Confidence (Secondary)")
    confidence = signals['confidence']
    st.metric("Confidence Score", f"{confidence}%")
    if confidence >= 75:
        st.success("🚀 High Probability Trade")
    elif confidence >= 60:
        st.info("👍 Moderate Probability")
    else:
        st.warning("⚠️ Low Confidence")

    st.subheader("🧠 Weighted Decision Engine (PRO, Secondary)")
    weighted_score = weighted_decision_engine(data, volume, ticker, timeframe)
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
    # ===== POSITION SIZING (DIPERBAIKI: ATR-based SL + peringatan risiko) =====
    st.subheader("📐 Position Sizing Calculator")
    atr_for_sl = safe_last(data['ATR'], 0.0)
    atr_sl_default = (last_close - ATR_STOP_MULT * atr_for_sl) if (atr_for_sl > 0 and last_close > ATR_STOP_MULT * atr_for_sl) else (last_close * 0.97 if last_close else 0.0)

    col_cap, col_risk, col_sl = st.columns(3)
    with col_cap:
        capital = st.number_input("Modal (Rp)", min_value=0.0, value=100_000_000.0, step=10_000_000.0)
    with col_risk:
        risk_percent = st.number_input("Risiko per Trade (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    with col_sl:
        use_atr_sl = st.checkbox(f"Gunakan ATR-based SL (Entry − {ATR_STOP_MULT:.0f}×ATR)",
                                 value=(atr_for_sl > 0),
                                 help="Disarankan: menyesuaikan stop loss dengan volatilitas historis saham.")
        if use_atr_sl and atr_for_sl > 0:
            stoploss_price = st.number_input("Stop Loss (Rp)", min_value=0.0,
                                             value=float(atr_sl_default), key="sl_atr_input")
            st.caption(f"ATR(14) = {atr_for_sl:.2f} → SL = {last_close:.2f} − {ATR_STOP_MULT:.0f}×{atr_for_sl:.2f} = {atr_sl_default:.2f}")
        else:
            stoploss_price = st.number_input("Stop Loss (Rp)", min_value=0.0,
                                             value=float(last_close * 0.97) if last_close else 0.0,
                                             key="sl_manual_input")

    # Peringatan risiko portofolio
    if risk_percent > MAX_RISK_PCT_WARNING:
        st.warning(
            f"⚠️ **Risiko per trade ({risk_percent}%) melebihi batas toleransi umum "
            f"({MAX_RISK_PCT_WARNING:.0f}% dari modal).** Pertimbangkan menurunkan risiko "
            f"atau memperbesar stop loss untuk menghindari kerugian besar pada satu transaksi."
        )

    if capital > 0 and risk_percent > 0 and stoploss_price > 0 and last_close > 0:
        risk_amount_calc = capital * (risk_percent / 100)
        price_risk = last_close - stoploss_price
        if price_risk > 0:
            suggested_shares = int(risk_amount_calc / price_risk)
            position_value = suggested_shares * last_close
            st.write(f"**Jumlah saham yang direkomendasikan:** {suggested_shares:,} lembar")
            st.write(f"**Risiko per transaksi:** Rp {risk_amount_calc:,.2f} ({risk_percent}% dari modal)")
            st.write(f"Nilai posisi: Rp {position_value:,.2f} ({position_value/capital*100:.1f}% dari modal)")
            if position_value > capital:
                st.error("❌ Nilai posisi melebihi modal! Turunkan jumlah saham atau perbesar stop loss.")
            elif position_value > capital * 0.5:
                st.warning("⚠️ Nilai posisi > 50% modal. Pertimbangkan diversifikasi.")
        else:
            st.warning("Stop loss harus di bawah harga saat ini.")
    else:
        st.info("Masukkan modal, risiko, dan stop loss untuk menghitung.")

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
                returns_opt = get_portfolio_returns(tuple(tickers_opt), period=period_opt)
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

# ========== TAB 5: BACKTEST (DIPERLUAS: MDD + Profit Factor + Win Rate) ==========
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
    commission = st.checkbox("Include Commission (0.1% per trade)", value=True,
                             help="Disarankan aktifkan untuk hasil backtest yang realistis.")
    commission_rate = 0.001 if commission else 0
    with st.spinner("Menjalankan backtest..."):
        bt_result = backtest_strategy(data, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
                                      sma_period=sma_period, commission=commission_rate)
    if bt_result:
        col_ret, col_trades, col_bh = st.columns(3)
        col_ret.metric("Total Return", f"{bt_result['total_return']:.2f}%")
        col_trades.metric("Jumlah Transaksi", bt_result['num_trades'])
        col_bh.metric("Buy & Hold Return", f"{bt_result['bh_return']:.2f}%")

        # ===== METRIK BARU: MDD, Profit Factor, Win Rate =====
        col_mdd, col_pf, col_wr = st.columns(3)
        col_mdd.metric(
            "Max Drawdown (MDD)", f"{bt_result['mdd']:.2f}%",
            help="Penurunan terbesar dari puncak ekuitas. Semakin kecil (mendekati 0), semakin baik."
        )
        pf_val = bt_result['profit_factor']
        pf_display = "∞" if pf_val == float('inf') else f"{pf_val:.2f}"
        col_pf.metric(
            "Profit Factor", pf_display,
            help="Total profit / total loss. >1 berarti profit lebih besar dari loss (profitable)."
        )
        col_wr.metric(
            "Win Rate", f"{bt_result['win_rate']:.1f}%",
            help="Persentase transaksi yang menghasilkan profit."
        )

        st.write(f"**Modal Akhir:** Rp {bt_result['final_capital']:,.2f}")

        # Interpretasi cepat
        if bt_result['mdd'] < -20:
            st.warning(f"⚠️ MDD {bt_result['mdd']:.2f}% cukup dalam. Strategi berisiko besar pada periode drawdown.")
        if pf_val != float('inf') and pf_val < 1.0 and bt_result['num_trades'] > 0:
            st.error(f"❌ Profit Factor {pf_val:.2f} < 1.0 — strategi merugi secara keseluruhan.")
        elif pf_val > 1.5:
            st.success(f"✅ Profit Factor {pf_val:.2f} > 1.5 — strategi profitable secara historis.")

        if not IS_CLOUD and len(bt_result['equity_curve']) > 0:
            st.subheader("📈 Equity Curve")
            eq_len = min(len(bt_result['equity_curve']), len(data.index))
            equity_df = pd.DataFrame({'Date': data.index[:eq_len], 'Equity': bt_result['equity_curve'][:eq_len]})
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Equity'], mode='lines', name='Equity'))
            fig_eq.update_layout(title="Pertumbuhan Modal", xaxis_title="Tanggal", yaxis_title="Nilai (Rp)", height=400)
            st.plotly_chart(fig_eq, use_container_width=True, config={'displayModeBar': False})
        with st.expander("Lihat Detail Transaksi"):
            if bt_result['trades']:
                st.dataframe(pd.DataFrame(bt_result['trades'], columns=['Tipe', 'Tanggal', 'Harga']))
            else:
                st.info("Tidak ada transaksi.")
        with st.expander("Lihat P&L per Transaksi"):
            if bt_result['realized_pnls']:
                pnl_df = pd.DataFrame({
                    'Trade #': range(1, len(bt_result['realized_pnls']) + 1),
                    'P&L (Rp)': [f"{p:,.0f}" for p in bt_result['realized_pnls']],
                    'Status': ['Profit' if p > 0 else 'Loss' if p < 0 else 'Flat'
                               for p in bt_result['realized_pnls']]
                })
                st.dataframe(pnl_df, use_container_width=True)
            else:
                st.info("Belum ada trade tertutup.")
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
            div_yield_val = fund.get('div_yield')
            if div_yield_val is not None and pd.notna(div_yield_val):
                if div_yield_val > 1:
                    div_str = f"{div_yield_val:.2f}%"
                else:
                    div_str = f"{div_yield_val*100:.2f}%"
            else:
                div_str = "N/A"
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PER (TTM)", f"{fund['pe']:.2f}" if pd.notna(fund['pe']) else "N/A")
                st.metric("PBV", f"{fund['pb']:.2f}" if pd.notna(fund['pb']) else "N/A")
                st.metric("Dividend Yield", div_str)
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
        **ATR (14)**: Average True Range — ukuran volatilitas harga. Dipakai untuk filter breakout & stop loss dinamis.  
        **Volume > Volume MA**: Volume di atas rata-rata  
        **AD > 0**: Akumulasi (tekanan beli)  
        **CMF > 0**: Tekanan beli  
        **Risk Reward Ratio**: Target/risk > 2 = good setup  
        **ATR-based Stop Loss**: SL = Entry − 2×ATR (menyesuaikan volatilitas saham).  
        **Confirmed Breakout**: Harga close > resistance + max(0.5% × resistance, 0.5×ATR). Menghindari fake breakout.  
        **Max Drawdown (MDD)**: Penurunan terbesar dari puncak ekuitas. Semakin kecil, semakin baik.  
        **Profit Factor**: Total profit / total loss. >1 = profitable.  
        **Win Rate**: Persentase transaksi profit.  
        **Ensemble AI Score**: Gabungan Rule (25%), ML (25%), Smart Money (20%), Macro (20%), Momentum Reward (10%)  
        **Rule Engine**: Trend-following (Close>SMA20, SMA20>SMA50, MACD>Signal) + bonus RSI<40 + konfirmasi volume. RSI<30 & harga<SMA20 → blacklist falling knife  
        **ML Model**: RandomForest yang dilatih dari data IHSG (global atau per ticker), dengan validasi train/test  
        **Smart Money**: Deteksi akumulasi/distribusi institusi  
        **Macro Filter**: Risk ON/OFF berdasarkan IHSG, USD/IDR, Nasdaq  
        **Momentum Reward**: Frekuensi hari naik + magnitude (Sharpe-like) dalam 20 hari terakhir  
        **Multi Timeframe**: Bullish jika harga > SMA20 di daily, weekly, monthly  
        **Portfolio Optimizer**: Mean-variance optimization untuk alokasi aset optimal  
        **PRIMARY vs SECONDARY**: Hanya Ensemble yang menjadi rekomendasi utama. Panel lain adalah konfirmasi sekunder.
        """)

# ========== TAB 7: SMART MONEY (DIPERBAIKI: status eksplisit) ==========
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
        elif smart_status == "Data Tidak Cukup":
            st.warning(
                "⚠️ **Data Tidak Cukup** — Saham ini memiliki <20 baris historis "
                "(kemungkinan IPO baru atau sangat illikuid). Smart Money tidak dapat dihitung andal."
            )
        else:
            st.info("Netral, belum ada sinyal kuat.")

        if smart_status != "Data Tidak Cukup":
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
        if len(ihsg) > 0:
            col1.metric("IHSG", f"{safe_last(ihsg):.2f}", f"{((safe_last(ihsg)/ihsg.iloc[0])-1)*100:.2f}%")
        else:
            col1.metric("IHSG", "N/A")
        if len(usd) > 0 and usd.iloc[0] != 0:
            col2.metric("USD/IDR", f"{safe_last(usd):.2f}", f"{((safe_last(usd)/usd.iloc[0])-1)*100:.2f}%")
        else:
            col2.metric("USD/IDR", "N/A")
        if len(nasdaq) > 0 and nasdaq.iloc[0] != 0:
            col3.metric("Nasdaq", f"{safe_last(nasdaq):.2f}", f"{((safe_last(nasdaq)/nasdaq.iloc[0])-1)*100:.2f}%")
        else:
            col3.metric("Nasdaq", "N/A")
    except Exception:
        st.info("Tidak dapat mengambil data makro saat ini.")

    # ===== CATATAN MAKRO LANJUTAN (BARU) =====
    st.divider()
    st.subheader("📌 Keterbatasan & Rekomendasi Makro Lanjutan")
    st.info(
        "💡 **Data Net Foreign Buy/Sell & BI Rate belum terintegrasi.** "
        "Filter makro saat ini hanya memakai IHSG, USD/IDR, dan Nasdaq. "
        "Untuk akurasi lebih tinggi pada saham lapis kedua/ketiga, pertimbangkan menambahkan:\n\n"
        "- **Net Foreign Flow harian** (dari IDX atau penyedia data pasar seperti RTI/Bloomberg) "
        "— sering menjadi market mover utama di BEI.\n"
        "- **BI Rate** (suku bunga acuan) — memengaruhi biaya modal & sektor perbankan/properti.\n"
        "- **Yield SUN 10Y** — indikator risk appetite investor asing.\n"
        "- **Commodity prices** (coal, CPO, nikel) — penting untuk sektor Mining/Energy."
    )

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
# ========== COPY SAMPAI SINI ==========
