# -*- coding: utf-8 -*-
"""
【量化 AI 預測決策網頁儀表板 - 波段起漲第一天過濾版】
更新重點：
1. 加入「波段初次觸發過濾 (First Trigger)」與「10 日訊號冷卻期」，消除連續重複箭頭
2. 頂部狀態列即時反映「今日是否為新發動買點 (NEW BUY)」或「多頭持倉續抱 (HOLD POSITION)」
3. 維持 0~100% 門檻動態拉動與雙 Y 軸特徵監控

【量化 AI 預測決策網頁儀表板 - SHAP 決策歸因與波段過濾版】
更新重點：
1. 整合 SHAP (TreeExplainer) 動態解析今日決策最大貢獻因子 (Top Drivers)
2. 新增 Tab 分頁視覺化呈現「今日特徵推拉力瀑布圖 (SHAP Waterfall)」
3. 保留波段起漲第一天標記、10 日冷卻期與 0~100% 門檻動態拉動

【量化 AI 預測決策網頁儀表板 - 高對比黃色買點與飽和漸層強化版】
1.BUY 起漲點顏色改為「高亮鮮黃色（#eab308 / #facc15）加上深色俐落外框」：
在紅綠 K 線與深色均線背景下具備極高的視覺對比度與凸顯效果。

2.特徵重要度排行顏色加深：自訂色彩漸層（從深青藍 #38bdf8 到高飽和深湛藍 #1d4ed8），徹底移除末端過淡發白的問題。

【量化 AI 預測決策網頁儀表板 - 完整買賣點與 SHAP 歸因版】
特色：
1. 整合「AI 波段買進點 (黃色向上箭頭)」與「策略動態賣出點 (洋紅向下箭頭)」
2. 支援側邊欄自訂停利 (%)、停損 (%) 與 20MA 破線出場開關
3. 整合 SHAP 單日決策歸因與特徵推拉力瀑布圖
4. 頂部狀態列與側邊診斷卡即時連動持倉與出場原因
【量化 AI 預測決策網頁儀表板 - 亮紫出場點與主圖切換版】
更新重點：
1. 出場點 (SELL) 改用高對比霓虹紫色 (#c084fc / #a855f7)，避免與紅 K 線混淆
2. 側邊欄新增「主圖呈現模式」：可一鍵切換「K 線 (Candlestick)」或「收盤價折線 (Line)」
3. 買進點維持高亮鮮黃色 (#facc15)，保留 SHAP 歸因與 20MA/60MA 均線

【量化 AI 預測決策網頁儀表板 - 點線收盤價、游標開關與精確進出場註解版】
1.收盤價線條改為點線（dot）：在折線模式下使用 dash="dot" 繪製。
2.側邊欄新增游標註解開關：提供「啟用游標懸停註解（Hover Tooltips）」選項，關閉時圖表維持乾淨無遮擋。
3.分流買賣懸停文字：
    * BUY 起漲點：顯示「進場原因」（預測突破機率與觸發標準）。
    * SELL 出場點：顯示「出場原因」（達標停利、觸發停損或破線）。
 
【量化 AI 預測決策網頁儀表板 - 單點獨立懸停嚴格分流版】
修正重點：
1. hovermode 改為 closest 獨立懸停，徹底解除 BUY/SELL 同時彈出的問題
2. 移動到黃色 BUY 箭頭：【只顯示進場原因】
3. 移動到紫色 SELL 箭頭：【只顯示出場原因】
4. 主圖背景線條不搶焦點，懸停乾淨俐落

【量化 AI 預測決策網頁儀表板 - 全圖層整合懸停註解版】
更新重點：
1. 游標移動時，整合顯示當日 K 線/收盤價、20MA/60MA、預測機率與監控特徵值
2. 遇到起漲點時額外顯示【進場原因】；遇到出場點時額外顯示【出場原因】
3. 支援側邊欄開關懸停註解與切換主圖呈現模式 (K 線 / 點線)

【量化 AI 預測決策網頁儀表板 - 單一動態註解嚴格分流版】
修正重點：
1. 游標移動時，若當天是起漲點則【只出現進場原因】，若是出場點則【只出現出場原因】，平時不出現任何原因
2. 保留 K 線/收盤點線、20MA/60MA、預測機率與監控特徵值的所有註解
3. 移除多餘的重疊 Trace 卡片，呈現乾淨精準的 hover 資訊

【量化 AI 預測決策網頁儀表板 - 實單交易損益與報酬率統計版】
特色：
1. 每次買進 1 張 (1,000 股)，精確計算回溯期間「已實現總獲利金額」、「總報酬率 (獲利/本金)」與「勝率」
2. 若當日持倉續抱，即時計算當前庫存「未實現浮動損益 (NTD & %)」
3. 右側決策診斷卡新增損益績效面板，出場點 hover 顯示單筆實現損益

【量化 AI 預測決策網頁儀表板 - 含台股手續費與證交稅真實損益版】
特色：
1. 側邊欄支援自訂「券商手續費折扣 (例如 6 折、2.8 折)」
2. 精確扣除買賣雙向手續費 (0.1425%) 與賣出證交稅 (0.3%)
3. 計算扣除真實摩擦成本後的「淨獲利金額」、「淨報酬率」與「當前未實現淨損益」
4. 預設 6.0 折版
執行方式: python -m streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st

# ==============================================================================
# 1. 頁面設定與自訂清爽淺色 CSS
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. 側邊欄：策略控制與進出場參數
# ==============================================================================
st.sidebar.markdown("### ⚙️ 策略控制面板")
stock_id = st.sidebar.text_input("股票代號", "2330")
model_type = st.sidebar.selectbox(
    "AI 模型架構", ["lightgbm", "xgboost", "catboost", "rf"]
)
chart_mode = st.sidebar.radio("主圖呈現模式", ["K 線 (Candlestick)", "收盤價點線 (Dot Line)"])

st.sidebar.markdown("#### 💬 提示註解控制")
show_hover = st.sidebar.checkbox("啟用游標懸停註解 (Hover Tooltips)", value=True)
days = st.sidebar.slider("回溯顯示天數", 30, 360, 120)

# ==============================================================================
# 3. 檔案檢查與讀取
# ==============================================================================
PREDICT_CSV = f"{stock_id}_predict_features.csv"
RAW_CSV = f"{stock_id}_raw_download.csv"
MODEL_BUNDLE = f"{stock_id}_{model_type}_model.pkl"

if not os.path.exists(PREDICT_CSV):
    st.error(f"❌ 找不到特徵檔 `{PREDICT_CSV}`，請先執行預測腳本！")
    st.stop()

if not os.path.exists(MODEL_BUNDLE):
    st.error(f"❌ 找不到模型檔 `{MODEL_BUNDLE}`，請先執行訓練腳本！")
    st.stop()

df_feat = pd.read_csv(PREDICT_CSV)
df_raw = pd.read_csv(RAW_CSV)

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

bundle = joblib.load(MODEL_BUNDLE)
model, scaler, feature_cols, default_threshold = (
    bundle["model"],
    bundle["scaler"],
    bundle["feature_cols"],
    bundle["threshold"],
)
default_threshold_pct = float(round(default_threshold * 100, 2))

# ==============================================================================
# 4. 側邊欄：進出場參數與特徵監控 (交易成本移至最底)
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
take_profit_pct = st.sidebar.slider("波段停利目標 (%)", 2.0, 20.0, 6.0, step=0.5)
stop_loss_pct = st.sidebar.slider("波段停損限制 (%)", 1.0, 10.0, 3.5, step=0.5)
use_ma20_exit = st.sidebar.checkbox("跌破 20MA 且機率力竭時出場", value=True)

feature_name_to_col = {
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
col_to_feature_name = {v: k for k, v in feature_name_to_col.items()}

selected_feat_name = st.sidebar.selectbox(
    "選擇監控特徵 (圖 3)", list(feature_name_to_col.keys())
)

# 【交易成本移至側邊欄最底部，預設 6.0 折】
st.sidebar.markdown("---")
st.sidebar.markdown("#### 💸 交易成本設定")
fee_discount = st.sidebar.number_input(
    "券商手續費折扣 (折數)",
    min_value=0.1,
    max_value=10.0,
    value=6.0,
    step=0.1,
    help="例如：6 折請輸入 6.0；2.8 折請輸入 2.8；無折扣請輸入 10.0",
)

# ==============================================================================
# 5. 模型推論、進出場模擬與台股真實扣稅損益計算 (每筆 1 張 = 1000 股)
# ==============================================================================
X_all = scaler.transform(df[feature_cols])
df["Prob"] = model.predict_proba(X_all)[:, 1]
df["Raw_Signal"] = (df["Prob"] >= current_threshold).astype(int)

df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

def calc_trading_cost(buy_val, sell_val, discount):
    fee_rate = 0.001425 * (discount / 10.0)
    buy_fee = max(20.0, buy_val * fee_rate)
    sell_fee = max(20.0, sell_val * fee_rate)
    tax = sell_val * 0.003
    return buy_fee + sell_fee + tax

buy_signals = np.zeros(len(df), dtype=int)
sell_signals = np.zeros(len(df), dtype=int)
buy_reasons = [""] * len(df)
sell_reasons = [""] * len(df)
holding_states = np.zeros(len(df), dtype=int)
trade_net_pnl = np.zeros(len(df), dtype=float)
trade_net_pnl_pct = np.zeros(len(df), dtype=float)

in_position = False
entry_price = 0.0
last_buy_idx = -999

for i in range(len(df)):
    close_p = df["Close"].iloc[i]
    ma20_p = df["MA20"].iloc[i]
    prob_p = df["Prob"].iloc[i]
    raw_sig = df["Raw_Signal"].iloc[i]
    
    # 買進判定
    if not in_position and raw_sig == 1:
        if (i - last_buy_idx >= 5):
            buy_signals[i] = 1
            in_position = True
            entry_price = close_p
            last_buy_idx = i
            holding_states[i] = 1
            buy_reasons[i] = f"AI 機率達標 ({prob_p*100:.1f}% ≥ {custom_threshold_pct:.1f}%)"
            continue
    
    # 出場判定
    if in_position:
        holding_states[i] = 1
        buy_total_val = entry_price * 1000
        sell_total_val = close_p * 1000
        total_costs = calc_trading_cost(buy_total_val, sell_total_val, fee_discount)
        
        net_profit_ntd = sell_total_val - buy_total_val - total_costs
        net_return_pct = net_profit_ntd / (buy_total_val + max(20.0, buy_total_val * 0.001425 * (fee_discount / 10.0)))
        gross_ret = (close_p - entry_price) / entry_price
        
        is_exit = False
        reason = ""
        
        if gross_ret >= (take_profit_pct / 100.0):
            is_exit = True
            reason = f"達標停利 (+{gross_ret*100:.1f}%)"
        elif gross_ret <= -(stop_loss_pct / 100.0):
            is_exit = True
            reason = f"觸發停損 ({gross_ret*100:.1f}%)"
        elif use_ma20_exit and (close_p < ma20_p) and (prob_p < current_threshold * 0.85):
            is_exit = True
            reason = "跌破 20MA 且動能轉弱"
            
        if is_exit:
            sell_signals[i] = 1
            sell_reasons[i] = f"{reason} | 淨損益: {net_profit_ntd:+,.0f} 元 ({net_return_pct*100:+.2f}%)"
            trade_net_pnl[i] = net_profit_ntd
            trade_net_pnl_pct[i] = net_return_pct
            in_position = False

df["Signal"] = buy_signals
df["Buy_Reason"] = buy_reasons
df["Sell_Signal"] = sell_signals
df["Sell_Reason"] = sell_reasons
df["Holding"] = holding_states
df["Trade_Net_PnL"] = trade_net_pnl
df["Trade_Net_PnL_Pct"] = trade_net_pnl_pct

plot_df = df.tail(days).copy().reset_index(drop=True)
latest = plot_df.iloc[-1]

# ==============================================================================
# 6. 回溯期累計淨損益與報酬率統計
# ==============================================================================
completed_trades = plot_df[plot_df["Sell_Signal"] == 1]
total_net_pnl_ntd = completed_trades["Trade_Net_PnL"].sum()
total_trades_count = len(completed_trades)
win_trades_count = (completed_trades["Trade_Net_PnL"] > 0).sum()
win_rate = (win_trades_count / total_trades_count * 100) if total_trades_count > 0 else 0.0

total_capital_deployed = 0.0
for idx in completed_trades.index:
    prior_buys = plot_df.loc[:idx][plot_df.loc[:idx]["Signal"] == 1]
    if not prior_buys.empty:
        cost_base = prior_buys.iloc[-1]["Close"] * 1000
        total_capital_deployed += (cost_base + max(20.0, cost_base * 0.001425 * (fee_discount / 10.0)))

total_net_roi_pct = (total_net_pnl_ntd / total_capital_deployed * 100) if total_capital_deployed > 0 else 0.0

unrealized_net_pnl_ntd = 0.0
unrealized_net_pnl_pct = 0.0
is_new_buy = latest["Signal"] == 1
is_new_sell = latest["Sell_Signal"] == 1
is_holding = latest["Holding"] == 1 and not is_new_sell

if is_holding:
    recent_buy = plot_df[plot_df["Signal"] == 1].iloc[-1]
    cost_val = recent_buy["Close"] * 1000
    curr_val = latest["Close"] * 1000
    est_costs = calc_trading_cost(cost_val, curr_val, fee_discount)
    unrealized_net_pnl_ntd = curr_val - cost_val - est_costs
    unrealized_net_pnl_pct = unrealized_net_pnl_ntd / (cost_val + max(20.0, cost_val * 0.001425 * (fee_discount / 10.0))) * 100

# ==============================================================================
# 7. SHAP 當日決策貢獻度動態計算
# ==============================================================================
shap_top_feature = "無"
shap_top_impact = 0.0
shap_df = pd.DataFrame()

try:
    explainer = shap.TreeExplainer(model)
    latest_scaled = X_all[-1:]
    raw_shap = explainer.shap_values(latest_scaled)

    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:
        shap_vals = raw_shap[0, :, 1]
    else:
        shap_vals = raw_shap[0]

    shap_df = pd.DataFrame({
        "特徵代碼": feature_cols,
        "特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols],
        "貢獻值(SHAP)": shap_vals,
        "原始數值": [latest[c] if c in latest else np.nan for c in feature_cols]
    }).sort_values(by="貢獻值(SHAP)", ascending=False).reset_index(drop=True)

    if not shap_df.empty:
        shap_top_feature = shap_df.iloc[0]["特徵名稱"]
        shap_top_impact = shap_df.iloc[0]["貢獻值(SHAP)"]
except Exception:
    shap_df = pd.DataFrame()

# ==============================================================================
# 8. 頂部狀態列
# ==============================================================================
if is_new_buy:
    signal_text = "⭐ 新觸發建倉 (BUY 1張)"
    delta_tag = "BUY TODAY"
    delta_color = "normal"
elif is_new_sell:
    signal_text = f"🟣 建議出場 ({latest['Sell_Reason']})"
    delta_tag = "SELL TODAY"
    delta_color = "inverse"
elif is_holding:
    signal_text = f"📈 多頭續抱中 (預估淨浮動: {unrealized_net_pnl_ntd:+,.0f} 元)"
    delta_tag = f"{unrealized_net_pnl_pct:+.2f}%"
    delta_color = "normal" if unrealized_net_pnl_ntd >= 0 else "inverse"
else:
    signal_text = "☕ 維持空手觀望 (WAIT)"
    delta_tag = "WAITING"
    delta_color = "off"

k1, k2, k3, k4 = st.columns(4)
k1.metric("基準交易日", str(latest["Date"]))
k2.metric("最新收盤報價", f"{latest['Close']:.1f} 元")
k3.metric("突破概率預測", f"{latest['Prob']*100:.2f}%", f"門檻 {custom_threshold_pct:.1f}%")
k4.metric("即時決策信號", signal_text, delta=delta_tag, delta_color=delta_color)

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 9. Tab 分頁佈局
# ==============================================================================
tab_main, tab_shap, tab_importance = st.tabs(
    ["📈 決策看板 (Interactive Dashboard)", "🔍 今日決策歸因 (SHAP Explanation)", "📊 全局特徵重要度 (Global Importance)"]
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
                f"【{stock_id}】{chart_title_prefix}、均線與 AI 買賣點 (每次1張)",
                f"動態預測機率 vs 門檻 ({custom_threshold_pct:.1f}%)",
                f"特徵監控: {selected_feat_name} (左軸) vs 收盤股價 (右軸)",
            ],
        )

        custom_main_hovers = []
        for _, r in plot_df.iterrows():
            lines = [f"<b>收盤:</b> {r['Close']:.1f} 元"]
            if "K" in chart_mode:
                lines.insert(0, f"<b>開:</b> {r['Open']:.1f} | <b>高:</b> {r['High']:.1f} | <b>低:</b> {r['Low']:.1f}")
            if r["Signal"] == 1:
                lines.append(f"<b style='color:#ca8a04;'>⭐ 進場原因:</b> {r['Buy_Reason']}")
            elif r["Sell_Signal"] == 1:
                lines.append(f"<b style='color:#7e22ce;'>🟣 出場原因:</b> {r['Sell_Reason']}")
            custom_main_hovers.append("<br>".join(lines))

        # 1. 主圖呈現
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

        # 20MA
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

        # 60MA
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

        # 買進訊號
        buys = plot_df[plot_df["Signal"] == 1].copy()
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["Date"],
                    y=(buys["Low"] if "K" in chart_mode else buys["Close"]) * 0.985,
                    mode="markers",
                    marker=dict(
                        symbol="arrow-up",
                        size=18,
                        color="#facc15",
                        line=dict(width=2.5, color="#1e293b"),
                    ),
                    name="BUY 訊號",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # 賣出訊號
        sells = plot_df[plot_df["Sell_Signal"] == 1].copy()
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["Date"],
                    y=(sells["High"] if "K" in chart_mode else sells["Close"]) * 1.015,
                    mode="markers",
                    marker=dict(
                        symbol="arrow-down",
                        size=18,
                        color="#c084fc",
                        line=dict(width=2.5, color="#581c87"),
                    ),
                    name="SELL 訊號",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # 2. 機率圖
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

        # 3. 特徵與收盤價
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
        st.markdown("#### 🧭 今日決策診斷")
        if is_new_buy:
            diag_box_color = "#eab308"
            diag_txt_color = "#ca8a04"
        elif is_new_sell:
            diag_box_color = "#c084fc"
            diag_txt_color = "#7e22ce"
        elif is_holding:
            diag_box_color = "#38bdf8"
            diag_txt_color = "#0369a1"
        else:
            diag_box_color = "#cbd5e1"
            diag_txt_color = "#0f172a"

        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {diag_box_color};">
                <div style="font-size: 0.85rem; color: #64748b;">模型推論結果</div>
                <div style="font-size: 1.25rem; font-weight: bold; color: {diag_txt_color}; margin: 4px 0;">
                    {signal_text}
                </div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    預測機率: <b>{latest['Prob']*100:.2f}%</b> (門檻: {custom_threshold_pct:.1f}%)
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px dashed #cbd5e1;" />
                <div style="font-size: 0.82rem; color: #475569;">
                    🥇 <b>最大驅動因子</b>: <br/>
                    <span style="color: #0284c7; font-weight: bold;">{shap_top_feature}</span> 
                    (推力: <code>+{shap_top_impact:.3f}</code>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ==============================================================================
        # 扣稅後淨損益與淨報酬率績效面板
        # ==============================================================================
        if fee_discount >= 10.0:
            fee_text = "(手續費+證交稅)"
        else:
            fee_text = f"({fee_discount:.1f}折手續費+證交稅)"

        st.markdown(f"#### 💰 淨損益統計 {fee_text}")
        pnl_color = "#dc2626" if total_net_pnl_ntd >= 0 else "#16a34a"
        pnl_sign = "+" if total_net_pnl_ntd >= 0 else ""
        
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {pnl_color};">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-size: 0.85rem;">已實現淨損益:</span>
                    <span style="font-weight: bold; color: {pnl_color}; font-size: 0.95rem;">
                        {pnl_sign}{total_net_pnl_ntd:,.0f} 元
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-size: 0.85rem;">累計淨報酬率:</span>
                    <span style="font-weight: bold; color: {pnl_color}; font-size: 0.95rem;">
                        {pnl_sign}{total_net_roi_pct:.2f}%
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-size: 0.85rem;">交易次數 / 勝率:</span>
                    <span style="font-weight: bold; color: #0f172a; font-size: 0.85rem;">
                        {total_trades_count} 次 ({win_rate:.1f}%)
                    </span>
                </div>
                <hr style="margin: 6px 0; border: none; border-top: 1px dashed #cbd5e1;" />
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b; font-size: 0.82rem;">當前庫存預估淨浮動:</span>
                    <span style="font-weight: bold; color: {'#dc2626' if unrealized_net_pnl_ntd>=0 else '#16a34a'}; font-size: 0.82rem;">
                        {'+' if unrealized_net_pnl_ntd>=0 else ''}{unrealized_net_pnl_ntd:,.0f} 元 ({unrealized_net_pnl_pct:+.2f}%)
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📋 即時因子監控")
        feat_display = []
        for name, col in feature_name_to_col.items():
            if col in latest:
                val = latest[col]
                if col == "inst_net_buy_20d":
                    val_str = f"{val/10_000_000:+.2f} 萬張"
                elif "mom" in col or "slope" in col or "bias" in col:
                    val_str = f"{val*100:+.2f}%"
                else:
                    val_str = f"{val:.2f}"
                feat_display.append({"因子名稱": name, "數值": val_str})

        st.dataframe(
            pd.DataFrame(feat_display),
            width="stretch",
            hide_index=True,
            height=400,
        )

with tab_shap:
    st.markdown(f"#### 🔍 【{latest['Date']}】當日決策 SHAP 因子推拉力拆解")
    st.caption("說明：綠色代表「強力推升買進機率」的加分因子；紅色代表「壓抑買進機率」的阻力因子。")

    if not shap_df.empty:
        shap_plot_df = shap_df.sort_values(by="貢獻值(SHAP)", ascending=True).copy()
        bar_colors = ["#16a34a" if val >= 0 else "#dc2626" for val in shap_plot_df["貢獻值(SHAP)"]]

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
            xaxis=dict(title="SHAP 貢獻值 (對 Log-Odds 影響力)", zeroline=True, zerolinecolor="#0f172a"),
            margin=dict(l=20, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_shap, width="stretch")

        st.markdown("##### 📋 當日前 5 大核心推手因子清單")
        st.dataframe(shap_df.head(5), width="stretch", hide_index=True)
    else:
        st.info("目前選擇的模型架構暫不支援 SHAP 解析。")

with tab_importance:
    st.markdown("#### 🧠 模型關鍵特徵重要度排行 (Top 10)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp_df = (
            pd.DataFrame({"特徵名稱": [col_to_feature_name.get(c, c) for c in feature_cols], "重要度權重": importances})
            .sort_values(by="重要度權重", ascending=True)
            .tail(10)
        )

        deep_blues = [
            [0.0, "#38bdf8"],
            [0.5, "#2563eb"],
            [1.0, "#1e3a8a"]
        ]

        fig_imp = go.Figure(
            go.Bar(
                x=feat_imp_df["重要度權重"],
                y=feat_imp_df["特徵名稱"],
                orientation="h",
                marker=dict(
                    color=feat_imp_df["重要度權重"],
                    colorscale=deep_blues,
                    showscale=False,
                    line=dict(color="#0f172a", width=0.8),
                ),
            )
        )
        fig_imp.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=460,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("目前選擇的模型架構不支援原生特徵重要度輸出。")