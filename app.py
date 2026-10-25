# retrain_global_model.py
import yfinance as yf
import pandas as pd
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn
print(f"scikit-learn: {sklearn.__version__}")

IHSG_BLUE_CHIPS = [
    "BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "ADRO.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", "ICBP.JK",
    "INDF.JK", "UNVR.JK", "SMGR.JK", "CPIN.JK", "JPFA.JK"
]

all_dfs = []
for t in IHSG_BLUE_CHIPS:
    try:
        df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=False)
        if df.empty or len(df) < 50:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna().copy()
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
        all_dfs.append(df.dropna())
    except Exception as e:
        print(f"Skip {t}: {e}")

data = pd.concat(all_dfs, ignore_index=True)
feature_cols = ['RSI','MACD','MACD_signal','SMA20','SMA50','Volume','Volume_MA','return_5d','volatility']
X = data[feature_cols].fillna(0)
y = data['target']
print(f"Training dengan {len(X)} baris, {y.nunique()} kelas")

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X, y)

joblib.dump({'model': model, 'feature_names': feature_cols}, 'global_model.joblib')
print("✅ global_model.joblib dibuat dengan sklearn", sklearn.__version__)
