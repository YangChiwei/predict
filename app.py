# -*- coding: utf-8 -*-
"""
【量化 AI 預測決策網頁儀表板 - 門檻 0~100% 自由調校與重設版】
執行方式: python -m streamlit run app.py
"""

import os
import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="台股量化 AI 決策看板", layout="wide")

# ----------------- 側邊欄：股票與模型選單 -----------------
st.sidebar.title("⚙️ 策略控制面板")
stock_id = st.sidebar.text_input("股票代號", "2330")
model_type = st.sidebar.selectbox(
    "選擇模型", ["lightgbm", "xgboost", "catboost", "rf"]
)
days = st.sidebar.slider("回溯顯示天數", 30, 360, 120)

# ----------------- 檔案檢查與讀取 -----------------
PREDICT_CSV = f"{stock_id}_predict_features.csv"
RAW_CSV = f"{stock_id}_raw_download.csv"
MODEL_BUNDLE = f"{stock_id}_{model_type}_model.pkl"

if not os.path.exists(PREDICT_CSV):
    st.error(
        f"❌ 找不到特徵檔 `{PREDICT_CSV}`，請先執行 `python 03_daily_predict._1_0_1.py`！"
    )
    st.stop()

if not os.path.exists(MODEL_BUNDLE):
    st.error(
        f"❌ 找不到模型檔 `{MODEL_BUNDLE}`，請先執行 `python 02_train_model_1_0_1.py --stock {stock_id} --model {model_type}`！"
    )
    st.stop()

df_feat = pd.read_csv(PREDICT_CSV)
df_raw = pd.read_csv(RAW_CSV)

# 統一轉字串對齊日期
df_feat["Date_str"] = pd.to_datetime(df_feat["Date"]).dt.strftime("%Y-%m-%d")
df_raw["Date_str"] = pd.to_datetime(df_raw["Date"]).dt.strftime("%Y-%m-%d")

merge_cols = ["Date_str", "Open", "High", "Low", "Close", "Volume"]
df = (
    pd.merge(
        df_feat,
        df_raw[[c for c in merge_cols if c in df_raw.columns]],
        on="Date_str",
        how="inner",
    )
    .sort_values("Date_str")
    .reset_index(drop=True)
)
df["Date"] = df["Date_str"]

# 載入模型推論
bundle = joblib.load(MODEL_BUNDLE)
model, scaler, feature_cols, default_threshold = (
    bundle["model"],
    bundle["scaler"],
    bundle["feature_cols"],
    bundle["threshold"],
)
default_threshold_pct = float(round(default_threshold * 100, 2))

# ----------------- 側邊欄：0~100% 門檻調整與重設機制 -----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 進場門檻調校")

# 使用 session_state 記錄與重設滑桿數值
threshold_key = f"thresh_{stock_id}_{model_type}"
if threshold_key not in st.session_state:
    st.session_state[threshold_key] = default_threshold_pct


def reset_threshold():
    st.session_state[threshold_key] = default_threshold_pct


# 重設按鈕
st.sidebar.button(
    f"🔄 回復原模型值 ({default_threshold_pct:.2f}%)",
    on_click=reset_threshold,
    use_container_width=True,
)

# 0% ~ 100% 滑桿
custom_threshold_pct = st.sidebar.slider(
    "自訂進場門檻 (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key=threshold_key,
    help="門檻範圍 0% ~ 100%，調整後上方 K 線的 BUY 訊號與 KPI 狀態會即時連動變更",
)
current_threshold = custom_threshold_pct / 100.0

feature_map = {
    "法人20日買超 (萬張)": "inst_net_buy_20d",
    "法人籌碼加速度": "inst_accel",
    "融資餘額月動能": "margin_mom",
    "月營收月增率": "revenue_mom",
    "20日均線斜率": "ma20_slope",
    "60日均線斜率": "ma60_slope",
    "20日乖離率": "bias_ma20",
    "60日乖離率": "bias_ma60",
    "5/20日均量比": "vol_ratio_5_20",
    "20日年化波動率": "volatility_20d",
    "RSI 3日差值": "rsi_diff_3d",
    "月級別 RSI": "rsi_monthly",
    "月級別 MACD": "macd_monthly",
    "台幣匯率20日月動能": "usd_twd_mom20",
    "VIX 20MA 乖離率": "vix_ma20_bias",
}
selected_feat_name = st.sidebar.selectbox(
    "選擇監控特徵 (圖 3)", list(feature_map.keys())
)

# 重新計算訊號
X_all = scaler.transform(df[feature_cols])
df["Prob"] = model.predict_proba(X_all)[:, 1]
df["Signal"] = (df["Prob"] >= current_threshold).astype(int)

plot_df = df.tail(days).copy()

# ----------------- 頂部 KPI 卡片 -----------------
latest = plot_df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("基準日期", str(latest["Date"]))
c2.metric("最新收盤價", f"{latest['Close']:.1f} 元")
c3.metric(
    "預測上漲機率",
    f"{latest['Prob']*100:.2f}%",
    delta=f"目前門檻: {custom_threshold_pct:.1f}%",
)
c4.metric(
    "AI 決策",
    "🔺 建議買進 (BUY)" if latest["Signal"] == 1 else "☕ 維持觀望 (HOLD)",
    delta="BUY" if latest["Signal"] == 1 else "HOLD",
)

st.markdown("---")

# ----------------- 互動圖表 (整合雙 Y 軸) -----------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.48, 0.24, 0.28],
    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
    subplot_titles=[
        f"【{stock_id}】K線走勢、20MA 與 BUY 買進訊號",
        f"預測機率時序圖 (當前門檻: {custom_threshold_pct:.1f}% | 預設: {default_threshold_pct:.1f}%)",
        f"特徵監控: {selected_feat_name} vs 收盤股價",
    ],
)

# 1. K 線圖與 20MA
fig.add_trace(
    go.Candlestick(
        x=plot_df["Date"],
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="K線",
        increasing_line_color="#d62728",
        decreasing_line_color="#2ca02c",
    ),
    row=1,
    col=1,
)

ma20 = plot_df["Close"].rolling(20).mean()
fig.add_trace(
    go.Scatter(
        x=plot_df["Date"],
        y=ma20,
        mode="lines",
        name="20MA",
        line=dict(color="#ff7f0e", width=1.5),
    ),
    row=1,
    col=1,
)

# 動態更新買進標記
buys = plot_df[plot_df["Signal"] == 1]
if not buys.empty:
    fig.add_trace(
        go.Scatter(
            x=buys["Date"],
            y=buys["Low"] * 0.985,
            mode="markers",
            marker=dict(symbol="triangle-up", size=13, color="crimson"),
            name="BUY 訊號",
        ),
        row=1,
        col=1,
    )

# 2. 機率圖與動態門檻虛線
fig.add_trace(
    go.Scatter(
        x=plot_df["Date"],
        y=plot_df["Prob"] * 100,
        mode="lines",
        name="預測機率 (%)",
        line=dict(color="#1f77b4", width=1.8),
    ),
    row=2,
    col=1,
)
fig.add_hline(
    y=custom_threshold_pct,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"自訂門檻 ({custom_threshold_pct:.1f}%)",
    row=2,
    col=1,
)

# 3. 最下方圖表：特徵 (左軸) + 收盤股價 (右軸)
feat_col = feature_map[selected_feat_name]
if feat_col in plot_df.columns:
    if feat_col == "inst_net_buy_20d":
        feat_vals = plot_df[feat_col] / 10_000_000
        colors = ["salmon" if v >= 0 else "lightgreen" for v in feat_vals]
        fig.add_trace(
            go.Bar(
                x=plot_df["Date"],
                y=feat_vals,
                name=selected_feat_name,
                marker_color=colors,
                opacity=0.75,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    else:
        feat_vals = plot_df[feat_col]
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=feat_vals,
                mode="lines",
                name=selected_feat_name,
                line=dict(color="#d62728", width=1.8),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

# 圖 3 右軸加入收盤價折線
fig.add_trace(
    go.Scatter(
        x=plot_df["Date"],
        y=plot_df["Close"],
        mode="lines",
        name="收盤價 (Close)",
        line=dict(color="#1f77b4", width=1.5, dash="dot"),
    ),
    row=3,
    col=1,
    secondary_y=True,
)

# 設定 Y 軸名稱
fig.update_yaxes(title_text=selected_feat_name, row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text="股價 (NTD)", row=3, col=1, secondary_y=True)

# 消除非交易日空白
fig.update_xaxes(type="category", row=1, col=1)
fig.update_xaxes(type="category", row=2, col=1)
fig.update_xaxes(type="category", row=3, col=1)

fig.update_layout(
    height=900,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=25, r=25, t=40, b=25),
)

st.plotly_chart(fig, use_container_width=True)