# -*- coding: utf-8 -*-
"""
終端機命令列執行

# 查看預設 (2330 + LightGBM)
python inspect_model.py
# 查看 2330 的 CatBoost 模型參數
python inspect_model.py --stock 2330 --model catboost
# 查看 2454 的 XGBoost 模型參數
python inspect_model.py --stock 2454 --model xgboost
# 查看 TFT 模型結構
python inspect_model.py --stock 2330 --model tft
"""
"""
在其他 Python 腳本或 Streamlit 中呼叫

from inspect_model import inspect_model_bundle

info = inspect_model_bundle(stock_id="2330", model_type="catboost")
print(info["best_params"])

"""
"""
【量化 AI 模型參數與資訊檢視工具】
用途：
- 載入指定股票代號與架構的模型包 (*_model.pkl)
- 顯示模型架構、最佳買進機率門檻、Optuna 最佳超參數、特徵欄位等詳細資訊
"""

import os
import argparse
import joblib
import json

ALL_MODELS = ["lightgbm", "xgboost", "catboost", "rf", "tft"]

def inspect_model_bundle(stock_id="2330", model_type="lightgbm"):
    """
    載入指定模型包並回傳模型資訊字典
    :param stock_id: 股票代號 (例: "2330")
    :param model_type: 模型類型 ("lightgbm", "xgboost", "catboost", "rf", "tft")
    :return: dict 包含模型詳細資訊
    """
    stock_id = str(stock_id).strip()
    model_type = model_type.lower().strip()
    model_bundle_name = f"{stock_id}_{model_type}_model.pkl"

    if not os.path.exists(model_bundle_name):
        raise FileNotFoundError(f"❌ 找不到模型檔案 `{model_bundle_name}`！請先確認是否已完成該模型的訓練。")

    bundle = joblib.load(model_bundle_name)

    info = {
        "stock_id": stock_id,
        "bundle_file": model_bundle_name,
        "model_type": bundle.get("model_type", model_type),
        "threshold": bundle.get("threshold", None),
        "best_params": bundle.get("best_params", {}),
        "feature_cols": bundle.get("feature_cols", []),
        "max_encoder_length": bundle.get("max_encoder_length", None),
        "ckpt_path": bundle.get("ckpt_path", None)
    }

    return info

def print_model_summary(info):
    """將模型資訊以易讀的格式輸出至終端機"""
    print("\n" + "=" * 60)
    print(f"📦 【{info['stock_id']}】模型結構與超參數總覽")
    print(f"📁 模型檔案路徑: {info['bundle_file']}")
    print("=" * 60)
    
    print(f"🔹 模型架構: {info['model_type'].upper()}")
    
    if info['threshold'] is not None:
        print(f"🎯 最佳買進機率門檻: {info['threshold']*100:.2f}% ({info['threshold']:.4f})")
    else:
        print("🎯 最佳買進機率門檻: 未記錄")

    if info.get("max_encoder_length"):
        print(f"⏱️ TFT 時序編碼長度: {info['max_encoder_length']} 天")

    print("\n⚙️ 【最佳超參數 (Best Hyperparameters)】:")
    if info['best_params']:
        # 以格式化 JSON 縮排呈現，閱讀更直觀
        print(json.dumps(info['best_params'], indent=4, ensure_ascii=False))
    else:
        print("  (此模型檔案未包含 best_params，可能使用舊版訓練程式產出)")

    print(f"\n📊 訓練特徵數量: {len(info['feature_cols'])} 個")
    print("=" * 60 + "\n")

# ==============================================================================
# 命令列獨立執行入口 (CLI)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="【台股量化 AI 模型參數檢視器】")
    parser.add_argument(
        "--stock", 
        type=str, 
        default="2330", 
        help="股票代號 (預設: 2330)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=ALL_MODELS,
        help=f"選擇要檢視的模型架構 (可選: {', '.join(ALL_MODELS)})",
    )

    args = parser.parse_args()

    try:
        model_info = inspect_model_bundle(stock_id=args.stock, model_type=args.model)
        print_model_summary(model_info)
    except FileNotFoundError as fnf_err:
        print(f"\n{fnf_err}\n")
    except Exception as e:
        print(f"\n❌ 讀取失敗: {e}\n")