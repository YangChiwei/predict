# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from build_features_01 import run_build_features
from train_model_02 import run_train_model
from daily_predict_03 import run_daily_predict
from ui_state import get_ui_stock, set_ui_stock

st.set_page_config(page_title="量化 AI 流水線管理", page_icon="⚙️", layout="wide")
st.title("⚙️ 台股量化 AI 流水線管理平台")

# 取得目前 UI 全域記憶的股票代號
current_stock = get_ui_stock()

tab1, tab2, tab3 = st.tabs([
    "📥 步驟 1：資料抓取與特徵工程",
    "🧠 步驟 2：多模型訓練與調優",
    "⚡ 步驟 3：增量更新與即時推論"
])

ALL_MODELS = ["lightgbm", "xgboost", "catboost", "rf", "tft"]

# ==================== TAB 1 ====================
with tab1:
    st.subheader("📥 歷史數據下載與特徵工程")
    
    with st.form("form_step1"):
        c1, c2, c3 = st.columns(3)
        s1_stock = c1.text_input("股票代號", value=current_stock, key="pipeline_stock_input")
        s1_start = c2.date_input("歷史起始日", pd.to_datetime("2018-01-01"), key="pipeline_start_date")
        s1_token = c3.text_input("FinMind Token (選填)", "", help="若有 FinMind 付費/專用 Token 可在此填入", key="pipeline_finmind_token")
        btn1 = st.form_submit_button(
            "🚀 下載並生成特徵", 
            width="stretch", 
            disabled=True, 
            help="雲端展示版限制：請於本機執行資料爬取"
        )

    if btn1:
        set_ui_stock(s1_stock) # 更新全域記憶
        with st.spinner("正在下載並計算特徵..."):
            try:
                raw_csv, train_csv, df = run_build_features(s1_stock, s1_start.strftime("%Y-%m-%d"), s1_token)
                st.success(f"🎉 產出成功！已儲存至 `{raw_csv}` 與 `{train_csv}`")
                st.dataframe(df.tail(8), width="stretch")
            except Exception as e:
                st.error(f"❌ 執行失敗: {e}")

# ==================== TAB 2 ====================
with tab2:
    st.subheader("🧠 模型超參數設定與訓練")

    c1, c2 = st.columns(2)
    def sync_s2_stock():
        set_ui_stock(st.session_state.s2_stock)
        
    s2_stock = c1.text_input("股票代號", value=current_stock, key="s2_stock", on_change=sync_s2_stock)
    s2_model = c2.selectbox("選擇 AI 模型架構", ALL_MODELS, key="s2_model")

    st.markdown("#### ⚙️ 調整模型起始超參數 (Hyperparameters)")

    custom_params = {}
    with st.container(border=True):
        if s2_model == "lightgbm":
            st.caption("設定 LightGBM 的 Optuna 搜尋範圍：")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                custom_params["n_estimators_min"] = st.number_input("最小樹木棵數 (n_estimators min)", 20, 100, 40, step=10)
                custom_params["n_estimators_max"] = st.number_input("最大樹木棵數 (n_estimators max)", 60, 300, 120, step=20)
            with col_b:
                custom_params["lr_min"] = st.number_input("學習率下限 (lr min)", 0.001, 0.05, 0.015, step=0.005, format="%.3f")
                custom_params["lr_max"] = st.number_input("學習率上限 (lr max)", 0.02, 0.2, 0.06, step=0.01, format="%.3f")
            with col_c:
                custom_params["max_depth_min"] = st.number_input("最小深度 (max_depth min)", 2, 4, 3)
                custom_params["max_depth_max"] = st.number_input("最大深度 (max_depth max)", 4, 8, 5)
                custom_params["num_leaves_min"] = 6
                custom_params["num_leaves_max"] = 18

        elif s2_model == "xgboost":
            st.caption("設定 XGBoost 的 Optuna 搜尋範圍：")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                custom_params["n_estimators_min"] = st.number_input("最小樹木棵數 (n_estimators min)", 20, 100, 40, step=10)
                custom_params["n_estimators_max"] = st.number_input("最大樹木棵數 (n_estimators max)", 60, 300, 120, step=20)
            with col_b:
                custom_params["lr_min"] = st.number_input("學習率下限 (lr min)", 0.001, 0.05, 0.015, step=0.005, format="%.3f")
                custom_params["lr_max"] = st.number_input("學習率上限 (lr max)", 0.02, 0.2, 0.06, step=0.01, format="%.3f")
            with col_c:
                custom_params["max_depth_min"] = st.number_input("最小深度 (max_depth min)", 2, 4, 3)
                custom_params["max_depth_max"] = st.number_input("最大深度 (max_depth max)", 4, 8, 5)

        elif s2_model == "catboost":
            st.caption("設定 CatBoost 的 Optuna 搜尋範圍：")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                custom_params["iterations_min"] = st.number_input("最小迭代次數 (iterations min)", 20, 100, 40, step=10)
                custom_params["iterations_max"] = st.number_input("最大迭代次數 (iterations max)", 60, 300, 120, step=20)
            with col_b:
                custom_params["lr_min"] = st.number_input("學習率下限 (lr min)", 0.001, 0.05, 0.015, step=0.005, format="%.3f")
                custom_params["lr_max"] = st.number_input("學習率上限 (lr max)", 0.02, 0.2, 0.06, step=0.01, format="%.3f")
            with col_c:
                custom_params["depth_min"] = st.number_input("最小深度 (depth min)", 2, 4, 3)
                custom_params["depth_max"] = st.number_input("最大深度 (depth max)", 4, 8, 5)

        elif s2_model == "rf":
            st.caption("設定 Random Forest (隨機森林) 的 Optuna 搜尋範圍：")
            col_a, col_b = st.columns(2)
            with col_a:
                custom_params["n_estimators_min"] = st.number_input("最小樹木棵數", 50, 200, 100, step=25)
                custom_params["n_estimators_max"] = st.number_input("最大樹木棵數", 150, 500, 300, step=50)
            with col_b:
                custom_params["max_depth_min"] = st.number_input("最小深度 (max_depth min)", 2, 3, 2)
                custom_params["max_depth_max"] = st.number_input("最大深度 (max_depth max)", 3, 8, 5)

        elif s2_model == "tft":
            st.caption("設定 Temporal Fusion Transformer (TFT) 神經網路結構：")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                custom_params["hidden_size"] = st.selectbox("隱藏層維度 (hidden_size)", [16, 32, 64], index=1)
                custom_params["attention_head_size"] = st.selectbox("注意力頭數 (attention_heads)", [2, 4, 8], index=1)
            with col_b:
                custom_params["learning_rate"] = st.number_input("學習率 (learning_rate)", 0.001, 0.05, 0.008, step=0.001, format="%.4f")
                custom_params["dropout"] = st.slider("Dropout 比例", 0.05, 0.5, 0.2, step=0.05)
            with col_c:
                custom_params["max_encoder_length"] = st.slider("時序編碼長度 (歷史回溯天數)", 20, 60, 40, step=5)

    c_opt1, c_opt2 = st.columns(2)
    s2_trials = c_opt1.slider("Optuna 搜尋迭代次數 (樹模型)", 10, 100, 30, step=5)
    s2_epochs = c_opt2.number_input("TFT 訓練 Epochs", 10, 100, 35, step=5)

    btn_train = st.button(
        "🔥 啟動訓練與超參數調優", 
        width="stretch", 
        type="primary", 
        disabled=True, 
        help="雲端算力限制：請於本機執行模型訓練"
    )

    if btn_train:
        set_ui_stock(s2_stock) # 同步
        with st.spinner(f"正在執行 {s2_model.upper()} 模型訓練與超參數調優，請稍候..."):
            try:
                res = run_train_model(s2_stock, s2_model, s2_trials, s2_epochs, custom_params)
                st.success(f"🎉 訓練完成！模型已儲存至 `{res['bundle_name']}`")

                m1, m2, m3, m4, m5 = st.columns(5)
                t_auc = res.get('test_auc')
                v_auc = res.get('val_auc')
                m1.metric("盲測集 ROC-AUC", f"{t_auc:.4f}" if t_auc is not None else "N/A")
                m2.metric("驗證集 ROC-AUC", f"{v_auc:.4f}" if v_auc is not None else "N/A")
                m3.metric("最佳進場門檻", f"{res['best_threshold']*100:.2f}%")
                m4.metric("驗證集準確率", f"{res['val_acc']*100:.2f}%")
                m5.metric("盲測集準確率", f"{res['test_acc']*100:.2f}%")

                with st.expander("🔍 點此查看【最佳超參數 (Best Hyperparameters)】", expanded=True):
                    st.json(res["best_params"])

                st.markdown("#### 📊 混淆矩陣 (Confusion Matrix)")
                cm_col1, cm_col2 = st.columns(2)

                def plot_cm(cm_data, title):
                    x_labels = ["預測: 觀望 (0)", "預測: 買進 (1)"]
                    y_labels = ["實際: 觀望 (0)", "實際: 買進 (1)"]
                    z_text = [[str(y) for y in x] for x in cm_data]
                    fig = ff.create_annotated_heatmap(
                        z=cm_data, x=x_labels, y=y_labels, annotation_text=z_text, colorscale="Blues", showscale=False
                    )
                    fig.update_layout(title=title, height=320, margin=dict(l=40, r=40, t=50, b=40))
                    return fig

                with cm_col1:
                    st.plotly_chart(plot_cm(res["cm_val"], f"驗證集 (Validation Set) - {s2_model.upper()}"), width="stretch")
                with cm_col2:
                    st.plotly_chart(plot_cm(res["cm_test"], f"盲測集 (Test Set) - {s2_model.upper()}"), width="stretch")

                st.markdown("#### 📋 盲測集分類報告 (Classification Report)")
                st.text(res["report"])
            except TimeoutError as te:
                st.warning(f"⚠️ 操作衝突：{te}")
            except Exception as e:
                st.error(f"❌ 訓練失敗: {e}")

# ==================== TAB 3 ====================
with tab3:
    st.subheader("⚡ 增量更新與每日自動預測")
    with st.form("form_step3"):
        c1, c2, c3 = st.columns(3)
        s3_stock = c1.text_input("股票代號", value=current_stock, key="pipeline_s3_stock")
        s3_model = c2.selectbox("調用模型", ALL_MODELS)
        s3_token = c3.text_input("FinMind Token (選填)", "", type="password")
        btn3 = st.form_submit_button(
            "🔄 執行增量推論", 
            width="stretch", 
            disabled=True, 
            help="雲端展示版限制：僅供檢視歷史推論結果"
        )

    if btn3:
        set_ui_stock(s3_stock) # 更新全域記憶
        with st.spinner("增量抓取最新數據並推論..."):
            try:
                date_str, prob, th, action, out_csv = run_daily_predict(s3_stock, s3_model, token=s3_token)
                st.markdown(f"### 🎯 今日 AI 決策：{'🚀【建議買進 / 建倉 (BUY)】' if action == 1 else '☕【維持觀望 / 不建倉 (HOLD)】'}")
                c1, c2, c3 = st.columns(3)
                c1.metric("基準日期", date_str)
                c2.metric("突破上漲機率", f"{prob*100:.2f}%")
                c3.metric("進場門檻", f"{th*100:.2f}%")
                st.success(f"✅ 即時推論檔已同步更新至 `{out_csv}`")
            except Exception as e:
                st.error(f"❌ 推論失敗: {e}")