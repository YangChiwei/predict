# -*- coding: utf-8 -*-
"""
【量化 AI 預測決策網頁儀表板 - 標題位置調整版】
調整重點：
1. 主圖標題移除「(每次1張)」文字。
2. 「每次 1 張」整合至右側「💰 淨損益統計 (每次1張, X.X折手續費+證交稅)」面板標題。
3. 維持第 2 頁滑動式日期選單與第 3 頁完整列出所有 15 個特徵值。

【量化 AI 預測決策網頁儀表板 - 股票名稱呈現與獲利因子連動版】
調整重點：
1. 自動辨識並呈現股票代號對應的中文/官方公司名稱（如 2303 聯電、2330 台積電、2317 鴻海）。
2. 主圖標題、側邊欄與指標卡片同步標註「代號 + 公司名稱」。
3. 移除爭議性的「累計淨報酬率 %」，改列專業量化標準「獲利因子 (Profit Factor)」。
4. 獲利因子零虧損保護（全勝無虧損時顯示 ∞）。
5. 支援第 2 頁買點下拉選單與日期滑動軸雙向自動對齊。
6. SHAP 貢獻長條圖採用台股紅漲（+）綠跌（-）色系。
7. 支援 CalibratedClassifierCV 模型解包，相容 SHAP 與特徵重要度解析。

【量化 AI 預測決策網頁儀表板】
1. [策略控制面板] 刪除英文註解文字，簡化畫面

【量化 AI 預測決策網頁儀表板 - TFT & 樹模型通用版】
1. 側邊欄模型選單擴充支援 TFT 模型架構
2. 統一載入與推論相容層，自動計算預測機率
3. Tab 2 (決策歸因)：支援樹模型 SHAP 與 TFT 原生變數貢獻度
4. Tab 3 (特徵重要度)：全面整合 amCharts 4 互動圖表

【量化 AI 預測決策網頁儀表板 - TFT & 樹模型通用版】
1. MariaDB

【量化 AI 預測決策網頁儀表板 - 自訂每筆投入本金 (萬元) 損益連動版】
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st
import yfinance as yf
from db_manager import read_from_db

try:
    import torch

    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

import warnings

# 針對 SHAP LightGBM TreeExplainer 的特定警告進行過濾
warnings.filterwarnings(
    "ignore", 
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"
)

# ==============================================================================
# 1. 頁面設定與自訂淺色 CSS
# ==============================================================================
st.set_page_config(
    page_title="台股量化 AI 決策終端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8fafc;
            color: #1e293b;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background: transparent !important;}
        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #edf2f7 !important;
            border-right: 1px solid #cbd5e1 !important;
        }
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #475569 !important; 
            font-size: 0.9rem !important; 
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important; 
            font-size: 1.4rem !important; 
            font-weight: 700 !important;
        }
        .stSidebar label, .stSidebar h3, .stSidebar h4 {
            color: #0f172a !important;
        }
        .stButton>button {
            background-color: #ffffff;
            color: #0284c7;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0284c7;
            color: #ffffff;
            border-color: #0284c7;
        }
        .info-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }
        .disclaimer-box {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 1.2rem;
            color: #991b1b;
            margin-bottom: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. 股票名稱對照工具
# ==============================================================================
COMMON_STOCK_NAMES = {
    "2330": "台積電", "2303": "聯電", "2317": "鴻海", "2454": "聯發科",
    "2308": "台達電", "2382": "廣達", "3231": "緯創", "2376": "技嘉",
    "2357": "華碩", "2412": "中華電", "2881": "富邦金", "2882": "國泰金",
    "2603": "長榮", "2609": "陽明", "2615": "萬海", "2002": "中鋼",
    "3008": "大立光", "2327": "國巨", "3034": "聯詠", "3037": "欣興",
    "6446": "藥華藥", "2345": "智邦", "3661": "世芯-KY", "6669": "緯穎"
}

@st.cache_data(ttl=86400)
def get_stock_name(stock_code):
    stock_code = str(stock_code).strip()
    if stock_code in COMMON_STOCK_NAMES:
        return COMMON_STOCK_NAMES[stock_code]
    try:
        ticker = yf.Ticker(f"{stock_code}.TW")
        info = ticker.info
        short_name = info.get("shortName") or info.get("longName")
        if short_name:
            return short_name
    except Exception:
        pass
    return ""

# ==============================================================================
# 3. 側邊欄控制面板
# ==============================================================================

from ui_state import get_ui_stock, set_ui_stock

st.sidebar.markdown("### ⚙️ 策略控制面板")

# 讀取全域記憶代號並綁定 on_change 觸發更新
current_stock = get_ui_stock()

def update_stock_from_input():
    set_ui_stock(st.session_state.stock_input)

stock_id = st.sidebar.text_input(
    "股票代號", 
    value=current_stock, 
    key="stock_input", 
    on_change=update_stock_from_input
)

stock_name = get_stock_name(stock_id)
display_title = f"{stock_id} {stock_name}".strip()


if stock_name:
    st.sidebar.markdown(
        f"""
        <div style="background-color: #e0f2fe; border: 1px solid #bae6fd; padding: 6px 12px; border-radius: 6px; margin-top: -6px; margin-bottom: 12px;">
            <span style="font-size: 0.85rem; color: #0369a1; font-weight: bold;">個股名稱：</span>
            <span style="font-size: 1.15rem; color: #0284c7; font-weight: 800;">{stock_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_type = st.sidebar.selectbox(
    "AI 模型架構", ["lightgbm", "xgboost", "catboost", "rf", "tft"]
)
chart_mode = st.sidebar.radio("主圖呈現模式", ["K 線", "收盤價點線"])

st.sidebar.markdown("#### 💬 提示註解控制")
show_hover = st.sidebar.checkbox("啟用游標懸停註解", value=True)
days = st.sidebar.slider("回溯顯示天數", 30, 360, 120)

# ==============================================================================
# 4. 從 MariaDB 載入數據與模型檢查
# ==============================================================================
MODEL_BUNDLE = f"{stock_id}_{model_type}_model.pkl"

if not os.path.exists(MODEL_BUNDLE):
    st.error(f"❌ 找不到模型檔 `{MODEL_BUNDLE}`，請先執行 02 訓練腳本！")
    st.stop()

query = """
    SELECT 
        f.*,
        r.open AS Open, r.high AS High, r.low AS Low, r.close AS Close, r.volume AS Volume
    FROM stock_features f
    JOIN stock_raw_data r ON f.stock_id = r.stock_id AND f.date = r.date
    WHERE f.stock_id = :sid
    ORDER BY f.date ASC
"""
df = read_from_db(query, {"sid": str(stock_id).strip()})

if df.empty:
    st.error(f"❌ MariaDB 內無股票 `{stock_id}` 之完整數據，請先在流水線管理中執行「步驟 1」！")
    st.stop()

df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df["Date_str"] = df["Date"]
bundle = joblib.load(MODEL_BUNDLE)
feature_cols = bundle["feature_cols"]
default_threshold = bundle["threshold"]
default_threshold_pct = float(round(default_threshold * 100, 2))

@st.cache_resource
def load_unified_model(m_type, _bundle_data, s_id):
    if m_type == "tft":
        ckpt_path = _bundle_data["ckpt_path"]
        max_enc = _bundle_data.get("max_encoder_length", 40)
        scaler = _bundle_data["scaler"]
        
        dummy_df = pd.DataFrame(0.0, index=np.arange(max_enc + 5), columns=feature_cols)
        dummy_df["time_idx"] = np.arange(len(dummy_df))
        dummy_df["stock_id"] = str(s_id)
        dummy_df["target_continuous"] = 0.0
        
        t_ds = TimeSeriesDataSet(
            dummy_df,
            time_idx="time_idx",
            target="target_continuous",
            group_ids=["stock_id"],
            min_encoder_length=max_enc,
            max_encoder_length=max_enc,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=feature_cols,
            target_normalizer=None,
            predict_mode=False,
        )
        t_model = TemporalFusionTransformer.from_dataset(
            t_ds,
            learning_rate=0.008,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.2,
            loss=QuantileLoss(),
            output_size=7,
        )
        t_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        t_model.eval()
        return {"type": "tft", "model": t_model, "scaler": scaler, "max_enc": max_enc, "importances": _bundle_data.get("feature_importances_")}
    else:
        return {"type": "tree", "model": _bundle_data["model"], "scaler": _bundle_data["scaler"]}

model_ctx = load_unified_model(model_type, bundle, stock_id)

# ==============================================================================
# 5. 側邊欄：進出場參數、資金與特徵監控
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 進場買進門檻")

threshold_key = f"thresh_{stock_id}_{model_type}"
if threshold_key not in st.session_state:
    st.session_state[threshold_key] = default_threshold_pct

def reset_threshold():
    st.session_state[threshold_key] = default_threshold_pct

st.sidebar.button(
    f"🔄 回復原模型值 ({default_threshold_pct:.2f}%)",
    on_click=reset_threshold,
    width="stretch",
)

custom_threshold_pct = st.sidebar.slider(
    "AI 買進機率門檻 (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key=threshold_key,
)
current_threshold = custom_threshold_pct / 100.0

st.sidebar.markdown("#### 🛡️ 出場賣出規則")

enable_take_profit = st.sidebar.checkbox("啟用波段停利目標", value=True)
if enable_take_profit:
    take_profit_pct = st.sidebar.slider("波段停利目標 (%)", 2.0, 200.0, 10.0, step=1.0)
else:
    take_profit_pct = None
    st.sidebar.caption("🔒 波段停利目標：已關閉 (讓獲利奔馳)")

stop_loss_pct = st.sidebar.slider("波段停損限制 (%)", 2.0, 20.0, 5.0, step=0.5)
use_ma20_exit = st.sidebar.checkbox("跌破 20MA 且機率力竭時出場", value=True)

feature_name_to_col = {
    "法人20日買超 (萬張)": "inst_net_buy_20d",
    "法人籌碼加速度": "inst_accel",
    "融資餘額月動能": "margin_mom",
    "月營收月增率": "revenue_mom",
    "20日均線斜率": "ma20_slope",
    "60日均線斜率": "ma60_slope",
    "20MA乖離率": "bias_ma20",
    "60MA乖離率": "bias_ma60",
    "5/20日均量比": "vol_ratio_5_20",
    "20日年化波動率": "volatility_20d",
    "3日RSI差值": "rsi_diff_3d",
    "月級別RSI": "rsi_monthly",
    "月級別MACD": "macd_monthly",
    "費城半導體動能": "sox_mom",
    "相對費半強弱度": "rs_vs_sox",
    "台幣匯率20日月動能": "usd_twd_mom20",
    "VIX 20MA 乖離率": "vix_ma20_bias",
}
col_to_feature_name = {v: k for k, v in feature_name_to_col.items()}

selected_feat_name = st.sidebar.selectbox(
    "選擇監控特徵", list(feature_name_to_col.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 💰 交易本金與手續費設定")

# ✨ 新增：輸入每筆買進本金（萬元為單位）
trade_capital_wan = st.sidebar.number_input(
    "每筆投入本金 (萬元)",
    min_value=1.0,
    max_value=10000.0,
    value=50.0,
    step=5.0,
    help="每次 AI 觸發買進訊號時，預計投入的總金額上限（單位：萬元）"
)
trade_capital_ntd = trade_capital_wan * 10000.0

fee_discount = st.sidebar.number_input(
    "券商手續費折扣 (折數)",
    min_value=0.1,
    max_value=10.0,
    value=6.0,
    step=0.1,
)

# ==============================================================================
# 6. 推論與回測邏輯 (支援本金換算股數)
# ==============================================================================
if model_ctx["type"] == "tft":
    tft_m = model_ctx["model"]
    scaler = model_ctx["scaler"]
    max_enc = model_ctx["max_enc"]
    
    infer_df = df.copy()
    if "time_idx" not in infer_df.columns:
        infer_df["time_idx"] = np.arange(len(infer_df))
    if "stock_id" not in infer_df.columns:
        infer_df["stock_id"] = str(stock_id)
    if scaler is not None:
        infer_df[feature_cols] = scaler.transform(infer_df[feature_cols]).astype(np.float32)
    infer_df["target_continuous"] = 0.0
    
    t_ds = TimeSeriesDataSet(
        infer_df,
        time_idx="time_idx",
        target="target_continuous",
        group_ids=["stock_id"],
        min_encoder_length=max_enc,
        max_encoder_length=max_enc,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=feature_cols,
        target_normalizer=None,
        predict_mode=False,
    )
    dl = t_ds.to_dataloader(batch_size=128, shuffle=False)
    with torch.no_grad():
        preds = tft_m.predict(dl, mode="prediction")
        raw_vals = preds.squeeze(-1).cpu().numpy()
    
    probs_1 = 1.0 / (1.0 + np.exp(-12.0 * (raw_vals - 0.02)))
    pad_len = len(df) - len(probs_1)
    raw_probs = np.concatenate([np.full(pad_len, fill_value=float(np.mean(probs_1[:20]))), probs_1])
else:
    #強制保證 C 連續記憶體）
    tree_m = model_ctx["model"]
    scaler = model_ctx["scaler"]
    # 直接傳入 DataFrame，對齊訓練時的特徵名稱
    X_all = np.ascontiguousarray(scaler.transform(df[feature_cols]), dtype=np.float32)
    raw_probs = tree_m.predict_proba(X_all)[:, 1]

df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

calibrated_probs = []
for idx in range(len(df)):
    c_p = df["Close"].iloc[idx]
    m20 = df["MA20"].iloc[idx]
    m60 = df["MA60"].iloc[idx]
    p = raw_probs[idx]
    if (c_p < m60) and (m20 < m60):
        calibrated_probs.append(0.0)
    elif (c_p > m20) and (m20 > m60):
        calibrated_probs.append(max(p, current_threshold * 1.05))
    else:
        calibrated_probs.append(p)

df["Prob"] = calibrated_probs
df["Raw_Signal"] = (df["Prob"] >= current_threshold).astype(int)

plot_df = df.tail(days).copy().reset_index(drop=True)
n_len = len(plot_df)

def calc_trading_cost(buy_val, sell_val, discount):
    fee_rate = 0.001425 * (discount / 10.0)
    buy_fee = max(20.0, buy_val * fee_rate)
    sell_fee = max(20.0, sell_val * fee_rate)
    tax = sell_val * 0.003
    return buy_fee + sell_fee + tax

buy_signals = np.zeros(n_len, dtype=int)
sell_signals = np.zeros(n_len, dtype=int)
buy_reasons = [""] * n_len
sell_reasons = [""] * n_len
holding_states = np.zeros(n_len, dtype=int)
holding_shares = np.zeros(n_len, dtype=float)  # ✨ 記錄各持倉點持股數
trade_net_pnl = np.zeros(n_len, dtype=float)
trade_net_pnl_pct = np.zeros(n_len, dtype=float)

in_position = False
entry_price = 0.0
position_shares = 0.0
last_buy_idx = -999

for i in range(n_len):
    close_p = plot_df["Close"].iloc[i]
    ma20_p = plot_df["MA20"].iloc[i]
    ma60_p = plot_df["MA60"].iloc[i]
    prob_p = plot_df["Prob"].iloc[i]
    raw_sig = plot_df["Raw_Signal"].iloc[i]
    
    is_bear = (close_p < ma60_p) and (ma20_p < ma60_p)
    if not in_position and raw_sig == 1 and not is_bear:
        if (i - last_buy_idx >= 5):
            buy_signals[i] = 1
            in_position = True
            entry_price = close_p
            # ✨ 依據自訂本金換算股數 (整數股)
            position_shares = int(trade_capital_ntd // close_p) if close_p > 0 else 0
            last_buy_idx = i
            holding_states[i] = 1
            holding_shares[i] = position_shares
            buy_reasons[i] = f"AI 機率達標 ({prob_p*100:.1f}%) | 買進 {position_shares:,} 股"
            continue
    
    if in_position:
        holding_states[i] = 1
        holding_shares[i] = position_shares
        gross_ret = (close_p - entry_price) / entry_price
        buy_total_val = entry_price * position_shares
        sell_total_val = close_p * position_shares
        costs = calc_trading_cost(buy_total_val, sell_total_val, fee_discount)
        net_profit_ntd = sell_total_val - buy_total_val - costs
        
        invested_base = buy_total_val + max(20.0, buy_total_val * 0.001425 * (fee_discount / 10.0))
        net_return_pct = net_profit_ntd / invested_base if invested_base > 0 else 0.0
        
        is_exit = False
        reason = ""
        
        if enable_take_profit and (take_profit_pct is not None) and (gross_ret >= (take_profit_pct / 100.0)):
            is_exit = True
            reason = f"達標停利 (+{gross_ret*100:.1f}%)"
        elif gross_ret <= -(stop_loss_pct / 100.0):
            is_exit = True
            reason = f"觸發停損 ({gross_ret*100:.1f}%)"
        elif use_ma20_exit and (close_p < ma20_p * 0.985) and (prob_p < current_threshold):
            is_exit = True
            reason = "跌破 20MA 趨勢終結"
            
        if is_exit:
            sell_signals[i] = 1
            trade_net_pnl[i] = net_profit_ntd
            trade_net_pnl_pct[i] = net_return_pct
            sell_reasons[i] = f"{reason} | 賣出 {position_shares:,} 股 | 淨損益: {net_profit_ntd:+,.0f} 元 ({net_return_pct*100:+.2f}%)"
            in_position = False
            position_shares = 0.0

plot_df["Signal"] = buy_signals
plot_df["Buy_Reason"] = buy_reasons
plot_df["Sell_Signal"] = sell_signals
plot_df["Sell_Reason"] = sell_reasons
plot_df["Holding"] = holding_states
plot_df["Holding_Shares"] = holding_shares
plot_df["Trade_Net_PnL"] = trade_net_pnl
plot_df["Trade_Net_PnL_Pct"] = trade_net_pnl_pct

latest = plot_df.iloc[-1]

# ==============================================================================
# 7. 回測統計數據 (連動自訂本金)
# ==============================================================================
completed_trades = plot_df[plot_df["Sell_Signal"] == 1]
total_net_pnl_ntd = completed_trades["Trade_Net_PnL"].sum()
total_trades_count = len(completed_trades)
win_trades_count = (completed_trades["Trade_Net_PnL"] > 0).sum()

gross_profit = completed_trades[completed_trades["Trade_Net_PnL"] > 0]["Trade_Net_PnL"].sum()
gross_loss = abs(completed_trades[completed_trades["Trade_Net_PnL"] < 0]["Trade_Net_PnL"].sum())

if gross_loss > 0:
    profit_factor_str = f"{gross_profit / gross_loss:.2f}"
elif gross_profit > 0 and gross_loss == 0:
    profit_factor_str = "∞ (全勝無虧損)"
else:
    profit_factor_str = "-"

unrealized_net_pnl_ntd = 0.0
unrealized_net_pnl_pct = 0.0
current_holding_shares = int(latest["Holding_Shares"])
is_new_buy = latest["Signal"] == 1
is_new_sell = latest["Sell_Signal"] == 1
is_holding = latest["Holding"] == 1 and not is_new_sell

if is_holding:
    recent_buys = plot_df[plot_df["Signal"] == 1]
    if not recent_buys.empty:
        recent_buy = recent_buys.iloc[-1]
        cost_val = recent_buy["Close"] * current_holding_shares
        curr_val = latest["Close"] * current_holding_shares
        est_costs = calc_trading_cost(cost_val, curr_val, fee_discount)
        unrealized_net_pnl_ntd = curr_val - cost_val - est_costs
        base_cost = cost_val + max(20.0, cost_val * 0.001425 * (fee_discount / 10.0))
        unrealized_net_pnl_pct = (unrealized_net_pnl_ntd / base_cost * 100) if base_cost > 0 else 0.0

# ==============================================================================
# 8. 歸因解析器 (SHAP 或 TFT 變數權重)
# ==============================================================================
def get_base_tree_model(m):
    if hasattr(m, "estimator") and m.estimator is not None:
        return m.estimator
    if hasattr(m, "calibrated_classifiers_") and len(m.calibrated_classifiers_) > 0:
        return m.calibrated_classifiers_[0].estimator
    return m

shap_top_feature = "無"
shap_top_impact = 0.0
shap_df_latest = pd.DataFrame()

if model_ctx["type"] == "tree":
    base_m = get_base_tree_model(model_ctx["model"])
    try:
        explainer = shap.TreeExplainer(base_m)
        latest_scaled = model_ctx["scaler"].transform(latest[feature_cols].to_frame().T)
        # 加上 check_additivity=False，阻斷 Apple Silicon 記憶體越界
        raw_s = explainer.shap_values(latest_scaled, check_additivity=False)
        if isinstance(raw_s, list):
            s_vals = raw_s[1][0]
        elif len(raw_s.shape) == 3:
            s_vals = raw_s[0, :, 1]
        else:
            s_vals = raw_s[0]
        shap_df_latest = pd.DataFrame({
            "特徵代碼": feature_cols,
            "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
            "貢獻值(SHAP)": s_vals,
        }).sort_values(by="貢獻值(SHAP)", ascending=False).reset_index(drop=True)
        shap_top_feature = shap_df_latest.iloc[0]["特徵名稱"]
        shap_top_impact = shap_df_latest.iloc[0]["貢獻值(SHAP)"]
    except Exception:
        explainer = None
else:
    explainer = None
    tft_imps = model_ctx.get("importances")
    if tft_imps is not None:
        shap_df_latest = pd.DataFrame({
            "特徵代碼": feature_cols,
            "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
            "貢獻值(SHAP)": tft_imps,
        }).sort_values(by="貢獻值(SHAP)", ascending=False).reset_index(drop=True)
        shap_top_feature = shap_df_latest.iloc[0]["特徵名稱"]
        shap_top_impact = shap_df_latest.iloc[0]["貢獻值(SHAP)"]

# ==============================================================================
# 9. 頂部狀態列
# ==============================================================================
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>免責聲明：</b>本程序所有 AI 模型預測、進出場訊號及損益回測數據均為量化演算法研究之用，<b>投資策略僅供參考，風險自負</b>。
    </div>
    """,
    unsafe_allow_html=True,
)

if is_new_buy:
    signal_text = f"⭐ 新觸發建倉 (買進約 {trade_capital_wan:.0f} 萬 / {current_holding_shares:,} 股)"
    delta_tag = "BUY TODAY"
    delta_color = "normal"
elif is_new_sell:
    signal_text = f"🟣 建議出場 ({latest['Sell_Reason']})"
    delta_tag = "SELL TODAY"
    delta_color = "inverse"
elif is_holding:
    signal_text = f"📈 多頭續抱中 ({current_holding_shares:,} 股，預估淨浮動: {unrealized_net_pnl_ntd:+,.0f} 元)"
    delta_tag = f"{unrealized_net_pnl_pct:+.2f}%"
    delta_color = "normal" if unrealized_net_pnl_ntd >= 0 else "inverse"
else:
    signal_text = "☕ 維持空手觀望 (WAIT)"
    delta_tag = "WAITING"
    delta_color = "off"

k1, k2, k3, k4 = st.columns(4)
k1.metric("股票代號/名稱/基準日", f"{display_title}", str(latest["Date"]))
k2.metric("最新收盤報價", f"{latest['Close']:.1f} 元")
k3.metric("突破概率預測", f"{latest['Prob']*100:.2f}%", f"門檻 {custom_threshold_pct:.1f}%")
k4.metric("即時決策信號", signal_text, delta=delta_tag, delta_color=delta_color)

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 10. Tab 分頁佈局
# ==============================================================================
tab_main, tab_shap, tab_importance = st.tabs(
    ["📈 決策看板 (Interactive Dashboard)", "🔍 決策歸因 (SHAP / TFT Attention)", "📊 全局特徵重要度 (Global Importance)"]
)

with tab_main:
    col_chart, col_side = st.columns([3.8, 1.2])

    with col_chart:
        chart_title_prefix = "K 線走勢" if "K" in chart_mode else "收盤價點線"
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.065,
            row_heights=[0.50, 0.24, 0.26],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],
            ],
            subplot_titles=[
                f"【{display_title}】{chart_title_prefix}、均線與 AI 買賣點",
                f"動態預測機率 vs 門檻 ({custom_threshold_pct:.1f}%)",
                f"特徵監控: {selected_feat_name} (左軸) vs 收盤股價 (右軸)",
            ],
        )

        custom_main_hovers = []
        for _, r in plot_df.iterrows():
            lines = [f"<b>{display_title}</b>", f"<b>收盤:</b> {r['Close']:.1f} 元"]
            if "K" in chart_mode:
                lines.insert(1, f"<b>開:</b> {r['Open']:.1f} | <b>高:</b> {r['High']:.1f} | <b>低:</b> {r['Low']:.1f}")
            if r["Signal"] == 1:
                lines.append(f"<b style='color:#ca8a04;'>⭐ 進場原因:</b> {r['Buy_Reason']}")
            elif r["Sell_Signal"] == 1:
                lines.append(f"<b style='color:#7e22ce;'>🟣 出場原因:</b> {r['Sell_Reason']}")
            custom_main_hovers.append("<br>".join(lines))

        if "K" in chart_mode:
            fig.add_trace(
                go.Candlestick(
                    x=plot_df["Date"],
                    open=plot_df["Open"],
                    high=plot_df["High"],
                    low=plot_df["Low"],
                    close=plot_df["Close"],
                    name="K線報價",
                    increasing_line_color="#dc2626",
                    increasing_fillcolor="#dc2626",
                    decreasing_line_color="#16a34a",
                    decreasing_fillcolor="#16a34a",
                    text=custom_main_hovers,
                    hoverinfo="text" if show_hover else "none",
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["Date"],
                    y=plot_df["Close"],
                    mode="lines",
                    name="收盤價",
                    line=dict(color="#0284c7", width=2.2, dash="dot"),
                    text=custom_main_hovers,
                    hovertemplate="%{text}<extra></extra>" if show_hover else None,
                    hoverinfo="none" if not show_hover else None,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["MA20"],
                mode="lines",
                name="20MA (月線)",
                line=dict(color="#f97316", width=1.5),
                hovertemplate="<b>20MA:</b> %{y:.1f} 元<extra></extra>" if show_hover else None,
                hoverinfo="none" if not show_hover else None,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["MA60"],
                mode="lines",
                name="60MA (季線)",
                line=dict(color="#6366f1", width=1.6),
                hovertemplate="<b>60MA:</b> %{y:.1f} 元<extra></extra>" if show_hover else None,
                hoverinfo="none" if not show_hover else None,
            ),
            row=1,
            col=1,
        )

        buys = plot_df[plot_df["Signal"] == 1].copy()
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["Date"],
                    y=(buys["Low"] if "K" in chart_mode else buys["Close"]) * 0.985,
                    mode="markers",
                    marker=dict(symbol="arrow-up", size=18, color="#facc15", line=dict(width=2.5, color="#1e293b")),
                    name="BUY 訊號",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        sells = plot_df[plot_df["Sell_Signal"] == 1].copy()
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["Date"],
                    y=(sells["High"] if "K" in chart_mode else sells["Close"]) * 1.015,
                    mode="markers",
                    marker=dict(symbol="arrow-down", size=18, color="#c084fc", line=dict(width=2.5, color="#581c87")),
                    name="SELL 訊號",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Prob"] * 100,
                mode="lines",
                name="預測機率",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
                showlegend=False,
                hovertemplate="<b>預測機率:</b> %{y:.2f}%<extra></extra>" if show_hover else None,
                hoverinfo="none" if not show_hover else None,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=custom_threshold_pct,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"門檻 {custom_threshold_pct:.1f}%",
            annotation_font_color="#ef4444",
            row=2,
            col=1,
        )

        feat_col = feature_name_to_col[selected_feat_name]
        if feat_col in plot_df.columns:
            if feat_col == "inst_net_buy_20d":
                feat_vals = plot_df[feat_col] / 10_000_000
                colors = ["#dc2626" if v >= 0 else "#16a34a" for v in feat_vals]
                fig.add_trace(
                    go.Bar(
                        x=plot_df["Date"],
                        y=feat_vals,
                        name=selected_feat_name,
                        marker_color=colors,
                        opacity=0.85,
                        showlegend=False,
                        hovertemplate="<b>法人買超:</b> %{y:+.2f} 萬張<extra></extra>" if show_hover else None,
                        hoverinfo="none" if not show_hover else None,
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
                        line=dict(color="#9333ea", width=1.8),
                        showlegend=False,
                        hovertemplate=f"<b>{selected_feat_name}:</b> %{{y:.2f}}<extra></extra>" if show_hover else None,
                        hoverinfo="none" if not show_hover else None,
                    ),
                    row=3,
                    col=1,
                    secondary_y=False,
                )

        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Close"],
                mode="lines",
                name="對照股價",
                line=dict(color="#64748b", width=1.4, dash="dot"),
                showlegend=False,
                hovertemplate="<b>對照收盤價:</b> %{y:.1f} 元<extra></extra>" if show_hover else None,
                hoverinfo="none" if not show_hover else None,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        fig.update_yaxes(title_text=selected_feat_name, row=3, col=1, secondary_y=False, showgrid=True, gridcolor="#e2e8f0")
        fig.update_yaxes(title_text="股價 (NTD)", row=3, col=1, secondary_y=True, showgrid=False, autorange=True)
        fig.update_xaxes(type="category", row=1, col=1)
        fig.update_xaxes(type="category", row=2, col=1)
        fig.update_xaxes(type="category", row=3, col=1)

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=900,
            xaxis_rangeslider_visible=False,
            hovermode="x unified" if show_hover else False,
            margin=dict(l=20, r=20, t=90, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
            ),
        )

        for annotation in fig["layout"]["annotations"]:
            annotation["font"] = dict(size=12, color="#334155")

        st.plotly_chart(fig, width="stretch")

    with col_side:
        st.markdown("###### 🧭 今日決策診斷")
        if is_new_buy:
            diag_box_color, diag_txt_color = "#eab308", "#ca8a04"
        elif is_new_sell:
            diag_box_color, diag_txt_color = "#c084fc", "#7e22ce"
        elif is_holding:
            diag_box_color, diag_txt_color = "#38bdf8", "#0369a1"
        else:
            diag_box_color, diag_txt_color = "#cbd5e1", "#0f172a"

        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {diag_box_color};">
                <div style="font-size: 0.85rem; color: #64748b;">【{display_title}】推論結果</div>
                <div style="font-size: 1.25rem; font-weight: bold; color: {diag_txt_color}; margin: 4px 0;">
                    {signal_text}
                </div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    預測機率: <b>{latest['Prob']*100:.2f}%</b> (門檻: {custom_threshold_pct:.1f}%)
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px dashed #cbd5e1;" />
                <div style="font-size: 0.82rem; color: #475569;">
                    🥇 <b>核心推手因子</b>: <br/>
                    <span style="color: #0284c7; font-weight: bold;">{shap_top_feature}</span> 
                    (推力: <code>+{shap_top_impact:.3f}</code>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fee_text = "(手續費+證交稅)" if fee_discount >= 10.0 else f"({fee_discount:.1f}折手續費+證交稅)"
        st.markdown(f"###### 💰 淨損益統計 (本金約 {trade_capital_wan:.0f} 萬, {fee_text})")

        loss_trades_count = total_trades_count - win_trades_count
        if total_trades_count > 0:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['勝場 (Profit)', '敗場 (Loss)'],
                values=[win_trades_count, loss_trades_count],
                hole=.55,
                domain={'x': [0.12, 0.88], 'y': [0.0, 1.0]},
                marker=dict(colors=['#dc2626', '#16a34a']),
                textinfo='label+percent',
                textfont_size=11,
                hoverinfo='label+value',
                showlegend=False
            )])
            pnl_sign = "+" if total_net_pnl_ntd >= 0 else ""
            fig_donut.update_layout(
                annotations=[dict(
                    text=f"已實現損益<br><b style='font-size:15px;color:#0f172a;'>{pnl_sign}{total_net_pnl_ntd:,.0f} 元</b>", 
                    x=0.5, y=0.5, font_size=11, showarrow=False
                )],
                margin=dict(l=25, r=25, t=10, b=10),
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("💡 目前區間尚無完成的交易歷史")

        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid #0284c7; padding: 12px; margin-top: -10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="color: #64748b;">獲利因子 (PF):</span>
                    <span style="font-weight: bold; color: #0284c7;">{profit_factor_str}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="color: #64748b;">總交易次數:</span>
                    <span style="font-weight: bold; color: #0f172a;">{total_trades_count} 次</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="color: #64748b;">當前持倉股數:</span>
                    <span style="font-weight: bold; color: #0f172a;">{current_holding_shares:,} 股</span>
                </div>
                <hr style="margin: 6px 0; border: none; border-top: 1px dashed #cbd5e1;" />
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span style="color: #64748b;">當前庫存預估淨浮動:</span>
                    <span style="font-weight: bold; color: {'#dc2626' if unrealized_net_pnl_ntd>=0 else '#16a34a'};">
                        {'+' if unrealized_net_pnl_ntd>=0 else ''}{unrealized_net_pnl_ntd:,.0f} 元 ({unrealized_net_pnl_pct:+.2f}%)
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("###### 📋 即時因子監控")
        feat_display = []
        for name, col in feature_name_to_col.items():
            if col in latest:
                val = latest[col]
                if col == "inst_net_buy_20d":
                    val_str = f"{val/10_000_000:+.2f} 萬張"
                elif "mom" in col or "slope" in col or "bias" in col or "regime" in col or "safe" in col:
                    val_str = f"{val*100:+.2f}%"
                else:
                    val_str = f"{val:.2f}"
                feat_display.append({"因子名稱": name, "數值": val_str})

        st.dataframe(pd.DataFrame(feat_display), width="stretch", hide_index=True, height=370)

# ==============================================================================
# TAB 2: 決策歸因 (SHAP / TFT Attention)
# ==============================================================================
with tab_shap:
    st.markdown(f"#### 🔍 【{display_title}】決策因子貢獻拆解")
    st.caption("說明：呈現各項總經與籌碼技術特徵在模型決策時的推拉力權重。")

    available_dates = plot_df["Date"].tolist()
    date_key = f"shap_selected_date_{stock_id}_{model_type}"

    if date_key not in st.session_state or st.session_state[date_key] not in available_dates:
        st.session_state[date_key] = available_dates[-1]

    buy_rows = plot_df[plot_df["Signal"] == 1]
    col_jump, col_info_tip = st.columns([1.5, 2.5])
    with col_jump:
        if not buy_rows.empty:
            buy_options = ["(手動選擇交易日)"] + [
                f"{r['Date']} | 收盤: {r['Close']:.1f}元 ({r['Buy_Reason']})" for _, r in buy_rows.iterrows()
            ]
            def on_buy_select():
                sel = st.session_state.get(f"buy_sel_{stock_id}_{model_type}")
                if sel and sel != "(手動選擇交易日)":
                    chosen_d = sel.split(" | ")[0]
                    if chosen_d in available_dates:
                        st.session_state[date_key] = chosen_d

            st.selectbox(
                "⭐ 快速跳轉至歷史買進起漲點 (BUY)：",
                options=buy_options,
                key=f"buy_sel_{stock_id}_{model_type}",
                on_change=on_buy_select,
            )
        else:
            st.info("💡 當前回溯區間內無買進訊號。")

    selected_date = st.select_slider(
        "📅 滑動選擇要拆解因子的交易日：",
        options=available_dates,
        key=date_key,
    )

    target_row = plot_df[plot_df["Date"] == selected_date].iloc[0]
    global_target_idx = df[df["Date"] == selected_date].index[0]

    selected_shap_df = pd.DataFrame()
    if model_ctx["type"] == "tree" and explainer is not None:
        target_scaled = model_ctx["scaler"].transform(df[feature_cols].iloc[global_target_idx : global_target_idx + 1])
        # 加上 check_additivity=False，阻斷 Apple Silicon 記憶體越界
        raw_s = explainer.shap_values(target_scaled, check_additivity=False)
        if isinstance(raw_s, list):
            t_shap = raw_s[1][0]
        elif len(raw_s.shape) == 3:
            t_shap = raw_s[0, :, 1]
        else:
            t_shap = raw_s[0]
            
        selected_shap_df = pd.DataFrame({
            "特徵代碼": feature_cols,
            "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
            "貢獻值(SHAP)": t_shap,
            "原始數值": [target_row[c] if c in target_row else np.nan for c in feature_cols]
        }).sort_values(by="貢獻值(SHAP)", ascending=False).reset_index(drop=True)
    else:
        tft_imps = model_ctx.get("importances")
        if tft_imps is not None:
            selected_shap_df = pd.DataFrame({
                "特徵代碼": feature_cols,
                "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
                "貢獻值(SHAP)": tft_imps,
                "原始數值": [target_row[c] if c in target_row else np.nan for c in feature_cols]
            }).sort_values(by="貢獻值(SHAP)", ascending=False).reset_index(drop=True)

    if not selected_shap_df.empty:
        shap_plot_df = selected_shap_df.sort_values(by="貢獻值(SHAP)", ascending=True).copy()
        bar_colors = ["#dc2626" if val >= 0 else "#16a34a" for val in shap_plot_df["貢獻值(SHAP)"]]

        fig_shap = go.Figure(
            go.Bar(
                x=shap_plot_df["貢獻值(SHAP)"],
                y=shap_plot_df["特徵名稱"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in shap_plot_df["貢獻值(SHAP)"]],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=540,
            xaxis=dict(title=f"【{selected_date}】因子權重 / 貢獻度", zeroline=True, zerolinecolor="#0f172a"),
            margin=dict(l=20, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_shap, width="stretch")

# ==============================================================================
# TAB 3: 全局特徵重要度 - amCharts 4
# ==============================================================================
with tab_importance:
    st.markdown(f"#### 🧠 【{display_title}】模型關鍵特徵重要度完整排行 (All Features)")

    importances = None
    if model_ctx["type"] == "tree":
        b_tree = get_base_tree_model(model_ctx["model"])
        if hasattr(b_tree, "feature_importances_"):
            importances = b_tree.feature_importances_
    else:
        importances = model_ctx.get("importances")

    if importances is not None:
        feat_imp_df = (
            pd.DataFrame({
                "特徵代碼": feature_cols,
                "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
                "重要度權重": importances
            })
            .sort_values(by="重要度權重", ascending=False)
            .reset_index(drop=True)
        )

        amcharts_data = [
            {
                "feature": str(row["特徵名稱"]),
                "importance": float(row["重要度權重"]),
                "code": str(row["特徵代碼"])
            }
            for _, row in feat_imp_df.iterrows()
        ]

        import json
        amcharts_json = json.dumps(amcharts_data, ensure_ascii=False)
        chart_height = max(500, len(amcharts_data) * 42 + 120)

        amcharts_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.amcharts.com/lib/4/core.js"></script>
<script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
<script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
<style>
    html, body {{
        margin: 0; padding: 0; width: 100%; background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #chartdiv {{ width: 100%; height: {chart_height}px; }}
</style>
</head>
<body>
<div id="chartdiv"></div>
<script>
am4core.useTheme(am4themes_animated);
var chart = am4core.create("chartdiv", am4charts.XYChart);
chart.padding(25, 45, 25, 20);
chart.data = {amcharts_json};

var categoryAxis = chart.yAxes.push(new am4charts.CategoryAxis());
categoryAxis.dataFields.category = "feature";
categoryAxis.renderer.minGridDistance = 1;
categoryAxis.renderer.inversed = true;
categoryAxis.renderer.grid.template.disabled = true;
categoryAxis.renderer.labels.template.fontSize = 13;
categoryAxis.renderer.labels.template.fill = am4core.color("#334155");

var valueAxis = chart.xAxes.push(new am4charts.ValueAxis());
valueAxis.min = 0;
valueAxis.renderer.labels.template.fontSize = 11;
valueAxis.renderer.labels.template.fill = am4core.color("#64748b");
valueAxis.renderer.grid.template.stroke = am4core.color("#e2e8f0");
valueAxis.title.text = "重要度權重";
valueAxis.title.fontSize = 12;

var series = chart.series.push(new am4charts.ColumnSeries());
series.dataFields.categoryY = "feature";
series.dataFields.valueX = "importance";
series.columns.template.strokeOpacity = 0;
series.columns.template.column.cornerRadiusBottomRight = 5;
series.columns.template.column.cornerRadiusTopRight = 5;
series.columns.template.height = am4core.percent(68);
series.tooltipText = "[bold]{{categoryY}}[/]\\n重要度: {{valueX.numberFormat('0.0000')}}\\n特徵代碼: {{code}}";

var labelBullet = series.bullets.push(new am4charts.LabelBullet());
labelBullet.label.horizontalCenter = "left";
labelBullet.label.dx = 10;
labelBullet.label.text = "{{values.valueX.numberFormat('0.0000')}}";
labelBullet.locationX = 1;
labelBullet.label.fontSize = 11;
labelBullet.label.fill = am4core.color("#334155");

series.columns.template.adapter.add("fill", function(fill, target) {{
    return chart.colors.getIndex(target.dataItem.index);
}});

chart.cursor = new am4charts.XYCursor();
</script>
</body>
</html>
"""
        st.iframe(amcharts_html, height=chart_height + 40, width="stretch")
    else:
        st.info("目前選擇的模型架構暫不支援特徵重要度輸出。")
