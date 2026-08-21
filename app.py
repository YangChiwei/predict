# -*- coding: utf-8 -*-
"""
【量化 AI 預測決策網頁儀表板 - 波段起漲第一天過濾版】
更新重點：
1. 加入「波段初次觸發過濾 (First Trigger)」與「10 日訊號冷卻期」，消除連續重複箭頭
2. 頂部狀態列即時反映「今日是否為新發動買點 (NEW BUY)」或「多頭持倉續抱 (HOLD POSITION)」
3. 維持 0~100% 門檻動態拉動與雙 Y 軸特徵監控

執行方式: python -m streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# 2. 側邊欄控制面板
# ==============================================================================
st.sidebar.markdown("### ⚙️ 策略控制面板")
stock_id = st.sidebar.text_input("股票代號", "2330")
model_type = st.sidebar.selectbox(
    "AI 模型架構", ["lightgbm", "xgboost", "catboost", "rf"]
)
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
# 4. 側邊欄：0~100% 門檻調校與重設
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 進場門檻調校")

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
    "自訂進場門檻 (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key=threshold_key,
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

# 進行推論計算
X_all = scaler.transform(df[feature_cols])
df["Prob"] = model.predict_proba(X_all)[:, 1]

# 原始達標訊號 (只要 >= 門檻即為 1)
df["Raw_Signal"] = (df["Prob"] >= current_threshold).astype(int)

# 【核心修正】：過濾連續密集訊號，只在「起漲第一天」或冷卻 10 天後發訊
cooldown_days = 10
filtered_signals = np.zeros(len(df), dtype=int)
last_signal_idx = -999

for i in range(len(df)):
    if df["Raw_Signal"].iloc[i] == 1:
        # 若前一天未發訊，或距離上次發訊已超過冷卻期，才判定為有效起漲買點
        if (i - last_signal_idx >= cooldown_days) or (i > 0 and df["Raw_Signal"].iloc[i - 1] == 0):
            if (i - last_signal_idx >= 5): # 至少間隔 5 天避免黏在一起
                filtered_signals[i] = 1
                last_signal_idx = i

df["Signal"] = filtered_signals

# 計算 20MA 與 60MA
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

plot_df = df.tail(days).copy()

# ==============================================================================
# 5. 頂部狀態列
# ==============================================================================
latest = plot_df.iloc[-1]
is_new_buy = latest["Signal"] == 1
is_in_trend = latest["Raw_Signal"] == 1

if is_new_buy:
    signal_text = "⬆️ 新觸發建倉 (NEW BUY)"
    delta_tag = "TRIGGERED TODAY"
    delta_color = "normal"
elif is_in_trend:
    signal_text = "📈 多頭持倉續抱 (HOLD POS)"
    delta_tag = "TREND CONTINUING"
    delta_color = "off"
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
# 6. Tab 分頁佈局
# ==============================================================================
tab_main, tab_importance = st.tabs(
    ["📈 決策看板 (Interactive Dashboard)", "📊 模型特徵重要度 (Feature Importance)"]
)

with tab_main:
    col_chart, col_side = st.columns([3.8, 1.2])

    with col_chart:
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
                f"【{stock_id}】K 線走勢、20MA 與 60MA (波段起點標記)",
                f"動態預測機率 vs 門檻 ({custom_threshold_pct:.1f}%)",
                f"特徵監控: {selected_feat_name} (左軸) vs 收盤股價 (右軸)",
            ],
        )

        # 1. 主 K 線 (紅漲綠跌)
        fig.add_trace(
            go.Candlestick(
                x=plot_df["Date"],
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="K線",
                increasing_line_color="#dc2626",
                increasing_fillcolor="#dc2626",
                decreasing_line_color="#16a34a",
                decreasing_fillcolor="#16a34a",
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
                line=dict(color="#d97706", width=1.5),
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
                line=dict(color="#8b5cf6", width=1.6),
            ),
            row=1,
            col=1,
        )

        # 買進訊號：只標記起漲第一天的有效買點
        buys = plot_df[plot_df["Signal"] == 1]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["Date"],
                    y=buys["Low"] * 0.985,
                    mode="markers",
                    marker=dict(
                        symbol="arrow-up",
                        size=17,
                        color="#0284c7",
                        line=dict(width=2, color="#ffffff"),
                    ),
                    name="BUY 起漲點",
                ),
                row=1,
                col=1,
            )

        # 2. 機率圖 (關閉圖例)
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Prob"] * 100,
                mode="lines",
                name="預測機率 (%)",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
                showlegend=False,
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

        # 3. 特徵 (左軸) + 收盤價 (右軸)
        feat_col = feature_map[selected_feat_name]
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
                name="收盤價 (Close)",
                line=dict(color="#64748b", width=1.4, dash="dot"),
                showlegend=False,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        # 座標軸設定
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
            hovermode="x unified",
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
        signal_box_color = "#0284c7" if is_new_buy else ("#38bdf8" if is_in_trend else "#cbd5e1")
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {signal_box_color};">
                <div style="font-size: 0.85rem; color: #64748b;">模型推論結果</div>
                <div style="font-size: 1.25rem; font-weight: bold; color: {'#0284c7' if is_new_buy else ('#0369a1' if is_in_trend else '#0f172a')}; margin: 4px 0;">
                    {signal_text}
                </div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    預測機率: <b>{latest['Prob']*100:.2f}%</b> (門檻: {custom_threshold_pct:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📋 即時因子監控")
        feat_display = []
        for name, col in feature_map.items():
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
            height=580,
        )

with tab_importance:
    st.markdown("#### 🧠 模型關鍵特徵重要度排行 (Top 10)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp_df = (
            pd.DataFrame({"特徵名稱": feature_cols, "重要度權重": importances})
            .sort_values(by="重要度權重", ascending=True)
            .tail(10)
        )

        fig_imp = go.Figure(
            go.Bar(
                x=feat_imp_df["重要度權重"],
                y=feat_imp_df["特徵名稱"],
                orientation="h",
                marker=dict(
                    color=feat_imp_df["重要度權重"],
                    colorscale="Blues",
                    showscale=False,
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