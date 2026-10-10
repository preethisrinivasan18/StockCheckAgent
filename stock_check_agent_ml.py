import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier


# ---------- CONFIG ----------
SHORT_WINDOW = 20
LONG_WINDOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

FEATURES = ["MA_gap", "RSI", "ret_1d", "ret_5d", "volatility_10d"]
ML_BUY_ABOVE = 0.55     # probability of UP above this -> BUY
ML_SELL_BELOW = 0.45    # probability of UP below this -> SELL
MIN_ROWS_FOR_ML = 120   # need enough history to train


# ---------- INDICATORS ----------
def add_moving_averages(df):
    df["MA_short"] = df["Close"].rolling(SHORT_WINDOW).mean()
    df["MA_long"] = df["Close"].rolling(LONG_WINDOW).mean()
    return df


def add_rsi(df):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_ml_features(df):
    """Numbers the ML model will learn from."""
    df["MA_gap"] = df["MA_short"] / df["MA_long"] - 1      # trend strength
    df["ret_1d"] = df["Close"].pct_change(1)                # 1-day change
    df["ret_5d"] = df["Close"].pct_change(5)                # 5-day change
    df["volatility_10d"] = df["Close"].pct_change().rolling(10).std()
    return df


# ---------- RULE-BASED SIGNAL ----------
def generate_signal(row):
    if pd.isna(row["MA_short"]) or pd.isna(row["MA_long"]) or pd.isna(row["RSI"]):
        return "NOT ENOUGH DATA"
    if row["MA_short"] > row["MA_long"] and row["RSI"] < RSI_OVERBOUGHT:
        return "BUY"
    elif row["MA_short"] < row["MA_long"] and row["RSI"] > RSI_OVERSOLD:
        return "SELL"
    return "HOLD"


# ---------- BACKTEST ----------
def backtest(df):
    df = df.copy()
    df["position"] = df["Signal"].shift(1).map({"BUY": 1, "HOLD": 0, "SELL": 0}).fillna(0)
    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["daily_return"] * df["position"]
    strategy_cum = (1 + df["strategy_return"]).cumprod().iloc[-1] - 1
    buy_hold_cum = (1 + df["daily_return"]).cumprod().iloc[-1] - 1
    return strategy_cum, buy_hold_cum


# ---------- ML MODEL ----------
def run_ml(df):
    """
    Train a Random Forest on the older 80% of the data,
    test it on the newest 20%, then predict for the latest day.
    Returns a dict, or None if there is not enough data.
    """
    data = df.copy()
    # Target: 1 if NEXT day's close is higher than today's, else 0
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    latest_row = data[FEATURES].iloc[[-1]]           # today (no answer known yet)
    train_data = data.dropna(subset=FEATURES).iloc[:-1]  # drop last day (no target)

    if latest_row.isna().any(axis=None) or len(train_data) < MIN_ROWS_FOR_ML:
        return None

    # Time-based split (never shuffle stock data - that would be cheating)
    split = int(len(train_data) * 0.8)
    train, test = train_data.iloc[:split], train_data.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42
    )
    model.fit(train[FEATURES], train["target"])

    accuracy = (model.predict(test[FEATURES]) == test["target"]).mean()
    baseline = max(test["target"].mean(), 1 - test["target"].mean())  # always guess majority
    prob_up = model.predict_proba(latest_row)[0][1]

    if prob_up > ML_BUY_ABOVE:
        ml_signal = "BUY"
    elif prob_up < ML_SELL_BELOW:
        ml_signal = "SELL"
    else:
        ml_signal = "HOLD"

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    return {
        "signal": ml_signal,
        "prob_up": prob_up * 100,
        "accuracy": accuracy * 100,
        "baseline": baseline * 100,
        "importance": importance,
    }

# ---------- DATA LOADING (cached so the site stays fast) ----------
@st.cache_data(ttl=3600)
def get_data(ticker, period):
    try:
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = add_moving_averages(data)
        data = add_rsi(data)
        data = add_ml_features(data)
        data["Signal"] = data.apply(generate_signal, axis=1)

        return data

    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None



@st.cache_data(ttl=3600)
def get_ml(ticker, period):
    df = get_data(ticker, period)
    if df is None:
        return None
    return run_ml(df)


# ---------- WEBSITE ----------
st.set_page_config(page_title="StockCheckAgent", layout="wide")
st.title("📈 StockCheckAgent")
st.caption("BUY / HOLD / SELL signals from Moving Average + RSI rules, and from a Machine Learning model")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect(
        "Choose stocks",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
         "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )
    period = st.selectbox("History period (ML needs 1y or more)", ["6mo", "1y", "2y", "5y"], index=2)

if not tickers:
    st.warning("Pick at least one stock from the sidebar.")
    st.stop()

# ----- Summary table -----
rows = []
for t in tickers:
    df = get_data(t, period)
    if df is None:
        continue
    latest = df.iloc[-1]
    strat_ret, bh_ret = backtest(df)
    ml = get_ml(t, period)
    rows.append({
        "Ticker": t,
        "Price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "Rule Signal": latest["Signal"],
        "ML Signal": ml["signal"] if ml else "NOT ENOUGH DATA",
        "ML Prob. Up %": ml["prob_up"] if ml else None,
        "ML Accuracy %": ml["accuracy"] if ml else None,
        "Strategy %": strat_ret * 100,
        "Buy&Hold %": bh_ret * 100,
    })

if not rows:
    st.error("No data found. Check your internet connection and try again.")
    st.stop()

summary_df = pd.DataFrame(rows)


def color_signal(val):
    return {
        "BUY": "background-color: #c6f6c6",
        "SELL": "background-color: #f6c6c6",
        "HOLD": "background-color: #f6f0c6",
    }.get(val, "")


st.subheader("Signal Summary")
styled = (
    summary_df.style
    .map(color_signal, subset=["Rule Signal", "ML Signal"])
    .format({
        "Price": "{:.2f}", "RSI": "{:.1f}", "ML Prob. Up %": "{:.1f}",
        "ML Accuracy %": "{:.1f}", "Strategy %": "{:.2f}", "Buy&Hold %": "{:.2f}",
    }, na_rep="-")
)
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(
    "ML Prob. Up % = chance the model gives that tomorrow's price is higher. "
    "ML Accuracy % = how often it was right on data it had never seen. "
    "Around 50-55% is normal for stock prediction - markets are very hard to predict."
)

# ----- Charts for one stock -----
st.subheader("Price Chart with Moving Averages")
selected = st.selectbox("Select stock to view chart", tickers)
chart_df = get_data(selected, period)

if chart_df is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name="Close Price"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_short"], name=f"MA {SHORT_WINDOW}"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_long"], name=f"MA {LONG_WINDOW}"))
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], name="RSI"))
    rsi_fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green")
    rsi_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(rsi_fig, use_container_width=True)

    # ----- ML details for the selected stock -----
    st.subheader(f"🤖 Machine Learning Details - {selected}")
    ml = get_ml(selected, period)
    if ml is None:
        st.info("Not enough history to train the model. Choose 1y, 2y or 5y in the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("ML Signal", ml["signal"])
        c2.metric("Chance price goes UP tomorrow", f"{ml['prob_up']:.1f}%")
        c3.metric("Model accuracy on unseen data", f"{ml['accuracy']:.1f}%",
                  delta=f"{ml['accuracy'] - ml['baseline']:.1f}% vs simple guessing")

        imp_fig = go.Figure(go.Bar(
            x=ml["importance"].values, y=ml["importance"].index, orientation="h"
        ))
        imp_fig.update_layout(
            title="Which numbers did the model rely on most?",
            height=300, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(imp_fig, use_container_width=True)

st.caption("For learning and demo only - this is not financial advice.")