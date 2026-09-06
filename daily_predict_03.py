# -*- coding: utf-8 -*-
"""
【模組三：增量更新與每日自動化預測 - 支援 5 種模型與 TFT】
支援模型: lightgbm | xgboost | catboost | rf | tft
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from FinMind.data import DataLoader
from sqlalchemy import text
from db_manager import get_engine, read_from_db

try:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

warnings.simplefilter(action="ignore", category=FutureWarning)


def run_daily_predict(stock_id="2330", model_type="lightgbm", start_date="2018-01-01", token=""):
    """
    執行增量數據下載、特徵更新與即時模型推論
    回傳: (prediction_date, prob, threshold, action, predict_csv_name)
    """
	
    model_type = model_type.lower()
    stock_id = str(stock_id).strip()
    model_bundle_name = f"{stock_id}_{model_type}_model.pkl"

    if not os.path.exists(model_bundle_name):
        raise FileNotFoundError(f"找不到模型檔案 {model_bundle_name}！請先執行訓練模組！")

    bundle = joblib.load(model_bundle_name)
    feature_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]

    # 1. 自動從 FinMind 與 yfinance 重新同步特徵 (調用步驟 1 寫入 MariaDB)
    from build_features_01 import run_build_features
    run_build_features(stock_id=stock_id, start_date=start_date, token=token)

    # 2. 從 MariaDB 讀取最新特徵進行推論
    query = "SELECT * FROM stock_features WHERE stock_id = :sid ORDER BY date ASC"
    predict_df = read_from_db(query, {"sid": stock_id})
    predict_df["Date"] = pd.to_datetime(predict_df["date"])
    predict_df["time_idx"] = np.arange(len(predict_df))

    # 3. 執行推論
    if model_type == "tft":
        # ... TFT 推論邏輯維持不變 ...[cite: 4]
        ckpt_path = bundle["ckpt_path"]
        max_encoder_length = bundle.get("max_encoder_length", 40)
        scaler = bundle["scaler"]
        infer_df = predict_df.copy()
        if scaler is not None:
            infer_df[feature_cols] = scaler.transform(infer_df[feature_cols]).astype(np.float32)
        infer_df["target_continuous"] = 0.0
        
        tft_dataset = TimeSeriesDataSet(
            infer_df, time_idx="time_idx", target="target_continuous",
            group_ids=["stock_id"], min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length, min_prediction_length=1,
            max_prediction_length=1, time_varying_unknown_reals=feature_cols,
            target_normalizer=None, predict_mode=False
        )
        tft_model = TemporalFusionTransformer.from_dataset(
            tft_dataset, learning_rate=0.008, hidden_size=32, attention_head_size=4,
            dropout=0.2, loss=QuantileLoss(), output_size=7
        )
        tft_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        tft_model.eval()
        dl = tft_dataset.to_dataloader(batch_size=128, shuffle=False)
        with torch.no_grad():
            preds = tft_model.predict(dl, mode="prediction")
            raw_val = preds[-1, 0].item()
        prob = 1.0 / (1.0 + np.exp(-12.0 * (raw_val - 0.02)))
    else:
        model, scaler = bundle["model"], bundle["scaler"]
        latest_row = predict_df.iloc[-1:]
        X_latest = latest_row[feature_cols]
        X_scaled = scaler.transform(X_latest)
        prob = float(model.predict_proba(X_scaled)[0, 1])

    latest_row = predict_df.iloc[-1:]
    prediction_date = pd.to_datetime(latest_row["Date"].values[0]).strftime("%Y-%m-%d")
    action = 1 if prob >= threshold else 0

    # 4. 將預測結果記錄至 MariaDB stock_predictions
    engine = get_engine()
    upsert_sql = """
        INSERT INTO stock_predictions (stock_id, date, model_type, probability, threshold, signal_action)
        VALUES (:sid, :dt, :mtype, :prob, :th, :act)
        ON DUPLICATE KEY UPDATE 
            probability = VALUES(probability),
            threshold = VALUES(threshold),
            signal_action = VALUES(signal_action);
    """
    with engine.begin() as conn:
        conn.execute(text(upsert_sql), {
            "sid": stock_id,
            "dt": prediction_date,
            "mtype": model_type,
            "prob": float(prob),
            "th": float(threshold),
            "act": int(action)
        })

    return prediction_date, prob, threshold, action, f"MariaDB: stock_predictions ({stock_id})"
# ==============================================================================
# 命令列獨立執行入口
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="【台股量化 AI 每日自動預測系統】")
    parser.add_argument("--stock", type=str, default="2330", help="股票代號")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "rf", "tft"],
        help="選擇調用的模型",
    )
    parser.add_argument("--start", type=str, default="2018-01-01", help="歷史起始日期")
    parser.add_argument("--token", type=str, default="", help="FinMind API Token")

    args = parser.parse_args()
    pred_date, p, th, act, csv_path = run_daily_predict(args.stock, args.model, args.start, args.token)
    print("\n" + "=" * 50)
    print(f"🎯 【{args.stock}】預測基準日: {pred_date} | 模型: {args.model.upper()}")
    print(f"📈 預測機率: {p * 100:.2f}% | 門檻: {th * 100:.2f}%")
    print(f"🚀 決策訊號: {'【建議買進 (BUY)】' if act == 1 else '【維持觀望 (HOLD)】'}")
    print("=" * 50 + "\n")