# -*- coding: utf-8 -*-
"""
【量化 AI 模型橫向評比與分析儀表板】
整合 benchmark_models 與 plot_csv 視覺化
新增檔案 benchmark_page.py，提供即時評比控制、歷史 CSV 載入與 Plotly 互動圖表
整合 plot_csv.py
"""

import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from benchmark_models import run_benchmark_process

st.set_page_config(page_title="模型自動化評比", page_icon="🏆", layout="wide")
st.title("🏆 多模型自動化評比與穩健度分析")

# ==========================================
# 1. 頂部控制面板 (整合 UI 跨頁同步記憶)
# ==========================================
from ui_state import get_ui_stock, set_ui_stock

# 讀取全域狀態
current_stock = get_ui_stock()

with st.expander("⚙️ 評比參數與即時執行設定", expanded=True):
    c1, c2, c3 = st.columns(3)
    
    # 綁定狀態變更機制
    def sync_bm_stock():
        set_ui_stock(st.session_state.bm_stock_id)
        
    stock_input = c1.text_input("股票代號", value=current_stock, key="bm_stock_id", on_change=sync_bm_stock)
    trials_input = c2.slider("Optuna 調優次數 (樹模型)", 10, 100, 30, step=5, key="bm_trials")
    epochs_input = c3.slider("TFT 訓練輪數 (Epochs)", 10, 100, 35, step=5, key="bm_epochs")

    selected_models = st.multiselect(
        "選擇要納入評比的模型",
        options=["lightgbm", "xgboost", "catboost", "rf", "tft"],
        default=["lightgbm", "xgboost", "catboost", "rf", "tft"]
    )

    btn_run = st.button(
    "🚀 啟動 5 大模型自動化評比", 
    type="primary", 
    width='stretch', 
    disabled=True, 
    help="雲端算力與記憶體限制：請於本機執行大量運算評比"
)

# 執行評比工作流程
if btn_run:
    if not selected_models:
        st.warning("⚠️ 請至少選擇一個模型進行評比！")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(curr, total, m_name, text):
            progress_bar.progress(curr / total)
            status_text.info(f"⏳ [{curr}/{total}] {text}")

        with st.spinner("正在自動化訓練與評估各模型，請稍候..."):
            try:
                df_res, csv_name = run_benchmark_process(
                    stock_id=stock_input,
                    n_trials=trials_input,
                    tft_epochs=epochs_input,
                    models_list=selected_models,
                    progress_callback=update_progress
                )
                progress_bar.progress(1.0)
                status_text.success(f"🎉 全數模型評比完成！報告已儲存至 `{csv_name}`")
                st.session_state["current_benchmark_df"] = df_res
                st.session_state["current_benchmark_file"] = csv_name
            except Exception as e:
                status_text.error(f"❌ 評比過程中斷: {e}")

# ==========================================
# 2. 歷史報告選擇或讀取當前結果
# ==========================================
st.markdown("---")
col_src1, col_src2 = st.columns([2, 1])

# 搜尋本機所有的 benchmark_results_*.csv
csv_files = sorted(glob.glob("benchmark_results_*.csv"), reverse=True)


selected_file = col_src1.selectbox(
    "📁 選擇評比歷史紀錄 CSV 檔案：",
    options=csv_files if csv_files else ["(查無本機評比檔案)"],
    index=0 if csv_files else 0
)

df_eval = None
if "current_benchmark_df" in st.session_state:
    df_eval = st.session_state["current_benchmark_df"]
elif selected_file and os.path.exists(selected_file):
    df_eval = pd.read_csv(selected_file)

# ==========================================
# 3. 視覺化圖表呈現 (整合 plot_csv.py)
# ==========================================
if df_eval is not None and not df_eval.empty:
    # 成功模型過濾
    success_df = df_eval[df_eval["Status"] == "Success"].copy()

    if success_df.empty:
        st.warning("⚠️ 載入的評比紀錄中無成功的訓練結果。")
    else:
        st.markdown("### 📊 評比視覺化洞察")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # 1. 買進勝率排行榜長條圖
            fig_bar = px.bar(
                success_df.sort_values(by="Test_Precision_BUY", ascending=True),
                x="Test_Precision_BUY",
                y="Model",
                orientation="h",
                color="Test_Precision_BUY",
                text=success_df.sort_values(by="Test_Precision_BUY", ascending=True)["Test_Precision_BUY"].apply(lambda v: f"{v*100:.2f}%"),
                title="🏆 盲測集「買進勝率 (Precision)」排行榜",
                labels={"Test_Precision_BUY": "買進勝率 (Precision)", "Model": "模型架構"},
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(height=380, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_bar, width='stretch')

        with chart_col2:
            # 2. 過擬合 vs 勝率 穩健度象限圖
            fig_scatter = px.scatter(
                success_df,
                x="Overfit_Gap",
                y="Test_Precision_BUY",
                text="Model",
                size="Test_Signals_Total",
                color="Test_Accuracy",
                title="🎯 模型穩健度象限 (左上方為「低過擬合 + 高勝率」最優區間)",
                labels={
                    "Overfit_Gap": "過擬合差距 (Val Acc - Test Acc)", 
                    "Test_Precision_BUY": "買進勝率 (Precision)",
                    "Test_Accuracy": "盲測準確率",
                    "Test_Signals_Total": "訊號總數"
                },
                color_continuous_scale="Blues"
            )
            # 加入零過擬合輔助線
            fig_scatter.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8")
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(height=380, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_scatter, width='stretch')

        # ==========================================
        # 4. 評比排行榜明細表格
        # ==========================================
        st.markdown("### 📋 評比指標總表 (Leaderboard)")
        
        # 格式化呈現
        styled_df = df_eval.copy()
        if "Test_Precision_BUY" in styled_df.columns:
            styled_df["Test_Precision_BUY"] = styled_df["Test_Precision_BUY"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
        if "Test_Accuracy" in styled_df.columns:
            styled_df["Test_Accuracy"] = styled_df["Test_Accuracy"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
        if "Val_Accuracy" in styled_df.columns:
            styled_df["Val_Accuracy"] = styled_df["Val_Accuracy"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
        if "Overfit_Gap" in styled_df.columns:
            styled_df["Overfit_Gap"] = styled_df["Overfit_Gap"].apply(lambda x: f"{x*100:+.2f}%" if pd.notnull(x) else "-")

        st.dataframe(styled_df, width='stretch', hide_index=True)
else:
    st.info("💡 尚未執行評比或找不到紀錄。請點擊上方按鈕開始評比，或確認工作目錄下是否有評比 CSV 檔案。")
