# -*- coding: utf-8 -*-
"""
【量化 AI 模型自動化測試與橫向評比腳本】
用途：
- 自動遍歷 5 大模型架構 (LightGBM, XGBoost, CatBoost, RF, TFT)
- 呼叫 train_model_02.py 執行超參數調優
- 記錄並解析各模型的 Val Acc, Test Acc, Precision, Recall, F1
- 匯出 benchmark_results.csv 並在終端機輸出比較排行榜

【量化 AI 模型自動化測試與橫向評比模組】
將評比邏輯封裝為 run_benchmark_process()，回傳 df_results 與 output_csv_name，並可傳入 Streamlit 的進度回調函式。
"""

import os
import re
import pandas as pd
from datetime import datetime
from train_model_02 import run_train_model

MODELS_TO_TEST = ["lightgbm", "xgboost", "catboost", "rf", "tft"]

def parse_classification_report(report_text):
    """從 sklearn classification_report 文字中提取 Class 1 (買進) 的 Precision, Recall, F1"""
    metrics = {"precision_1": 0.0, "recall_1": 0.0, "f1_1": 0.0}
    for line in report_text.splitlines():
        line = line.strip()
        if re.match(r"^1(?:\.0)?\s+", line):
            parts = [p for p in line.split() if p]
            if len(parts) >= 4:
                metrics["precision_1"] = float(parts[1])
                metrics["recall_1"] = float(parts[2])
                metrics["f1_1"] = float(parts[3])
            break
    return metrics

def run_benchmark_process(stock_id="2330", n_trials=30, tft_epochs=35, models_list=None, progress_callback=None):
    """
    執行 5 大模型自動化調優評比
    :param stock_id: 股票代號
    :param n_trials: 樹模型 Optuna 搜尋次數
    :param tft_epochs: TFT 訓練輪數
    :param models_list: 要測試的模型清單
    :param progress_callback: 回呼函式 callback(current_idx, total, model_name, status_text)
    :return: (pd.DataFrame, str output_csv)
    """
    models = models_list or MODELS_TO_TEST
    stock_id = str(stock_id).strip()
    output_csv = f"benchmark_results_{stock_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    results = []

    for idx, model_name in enumerate(models, 1):
        if progress_callback:
            progress_callback(idx, len(models), model_name, f"正在訓練與調優 【{model_name.upper()}】...")

        start_time = datetime.now()
        try:
            res = run_train_model(
                stock_id=stock_id,
                model_type=model_name,
                n_trials=n_trials,
                tft_epochs=tft_epochs
            )
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            rep_metrics = parse_classification_report(res.get("report", ""))

            cm_test = res.get("cm_test", [[0, 0], [0, 0]])
            tp = cm_test[1][1] if len(cm_test) > 1 else 0
            fp = cm_test[0][1] if len(cm_test) > 1 else 0
            fn = cm_test[1][0] if len(cm_test) > 1 else 0
            tn = cm_test[0][0] if len(cm_test) > 1 else 0

            # ✨ 成功時：抓取回傳的 test_auc 與 val_auc
            record = {
                "Stock_ID": stock_id,
                "Model": model_name.upper(),
                "Best_Threshold": round(res["best_threshold"], 4),
                "Test_AUC": round(res.get("test_auc") if res.get("test_auc") is not None else 0.5, 4),
                "Val_AUC": round(res.get("val_auc") if res.get("val_auc") is not None else 0.5, 4),
                "Val_Accuracy": round(res["val_acc"], 4),
                "Test_Accuracy": round(res["test_acc"], 4),
                "Overfit_Gap": round(res["val_acc"] - res["test_acc"], 4),
                "Test_Precision_BUY": rep_metrics["precision_1"],
                "Test_Recall_BUY": rep_metrics["recall_1"],
                "Test_F1_BUY": rep_metrics["f1_1"],
                "Test_Signals_Total": tp + fp,
                "True_Positives": tp,
                "Elapsed_Sec": round(elapsed_sec, 1),
                "Status": "Success"
            }
        except Exception as e:
            # ✨ 失敗時：同樣補上 Test_AUC 與 Val_AUC 預設值 0.0
            record = {
                "Stock_ID": stock_id,
                "Model": model_name.upper(),
                "Best_Threshold": 0.0,
                "Test_AUC": 0.0,  # ✨ 補上
                "Val_AUC": 0.0,   # ✨ 補上
                "Val_Accuracy": 0.0,
                "Test_Accuracy": 0.0,
                "Overfit_Gap": 0.0,
                "Test_Precision_BUY": 0.0,
                "Test_Recall_BUY": 0.0,
                "Test_F1_BUY": 0.0,
                "Test_Signals_Total": 0,
                "True_Positives": 0,
                "Elapsed_Sec": 0.0,
                "Status": f"Failed: {str(e)}"
            }

        results.append(record)

    df_results = pd.DataFrame(results)
    if "Test_Precision_BUY" in df_results.columns:
        df_results = df_results.sort_values(
            by=["Test_Precision_BUY", "Test_Accuracy"], 
            ascending=[False, False]
        ).reset_index(drop=True)

    df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df_results, output_csv

if __name__ == "__main__":
    df, csv_file = run_benchmark_process()
    print(df.to_string(index=False))