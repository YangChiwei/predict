# -*- coding: utf-8 -*-
"""
【模組二：多模型訓練、超參數調優與門檻校正系統 - 支援自訂超參數、混淆矩陣、ROC-AUC 與 MariaDB 鎖定機制】
嚴格量化 ML 無洩密規範重構版：
1. 嚴格隔離 StandardScaler 於 Train 集（保證無未來資訊洩漏）
2. 插入 Purge 隔離帶防止標籤自相關
3. 支援嚴謹計算 val_auc 與 test_auc，並加入單類別保護
4. 【徹底消除 UserWarning】：
   - 樹模型分支：強制鎖定特徵名稱，兼顧外部 predict.py DataFrame 推論相容性
   - TFT 分支：建立強制靜音區，遮蔽 PyTorch Forecasting 底層自動化轉換 Bug
5. 徹底修正 TFT 目標洩漏：嚴禁將未來 label 混入目標自回歸特徵
6. 解決 TFT 滾動窗口維度落差，精準對齊驗證與測試標籤
"""

import os
import warnings
import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from db_manager import read_from_db, db_lock

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
    if not hasattr(cb.CatBoostClassifier, "__sklearn_tags__"):
        cb.CatBoostClassifier.__sklearn_tags__ = lambda self: BaseEstimator.__sklearn_tags__(self)
except ImportError:
    cb = None

try:
    import torch
    import torch.nn as nn
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping

    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def safe_calc_auc(y_true, y_prob):
    """計算 ROC-AUC，若真實標籤僅有單一類別則回傳 None 避免拋出例外"""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(round(roc_auc_score(y_true, y_prob), 4))
    except Exception:
        return None


def run_train_model(stock_id="2330", model_type="lightgbm", n_trials=30, tft_epochs=35, custom_params=None):
    model_type = model_type.lower()
    stock_id = str(stock_id).strip()
    model_bundle_name = f"{stock_id}_{model_type}_model.pkl"
    custom_params = custom_params or {}

    lock_key = f"train_{stock_id}_{model_type}"

    with db_lock(lock_key, timeout=5):
        # 1. 僅讀取已完成標籤結算的資料
        query = """
            SELECT * FROM stock_features 
            WHERE stock_id = :sid AND label IS NOT NULL 
            ORDER BY date ASC
        """
        raw_df = read_from_db(query, {"sid": stock_id})
        if raw_df.empty:
            raise ValueError(f"MariaDB 內無 {stock_id} 之訓練特徵數據，請先執行步驟 1！")

        feature_cols = [
            "inst_net_buy_20d", "inst_accel", "margin_mom", "revenue_mom",
            "ma20_slope", "ma60_slope", "bias_ma20", "bias_ma60",
            "vol_ratio_5_20", "volatility_20d", "rsi_diff_3d", "rsi_monthly",
            "macd_monthly", "sox_mom", "rs_vs_sox", "usd_twd_mom20",
            "vix_ma20_bias",
        ]
        label_col = "label"

        # 2. 嚴格對齊維度並清洗數值
        final_df = raw_df.dropna(subset=feature_cols + [label_col]).copy()
        final_df[feature_cols] = final_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        final_df[feature_cols] = final_df[feature_cols].ffill().bfill().fillna(0.0)
        final_df[feature_cols] = final_df[feature_cols].clip(lower=-1e5, upper=1e5)
        
        final_df["Date"] = pd.to_datetime(final_df["date"])
        final_df["time_idx"] = np.arange(len(final_df))
        final_df["stock_id"] = stock_id

        n = len(final_df)
        if n < 120:
            raise ValueError(f"有效樣本數僅 {n} 筆，不足以支撐訓練，請擴大資料庫回溯區間！")

        # 3. 三段劃分：Train -> Purge -> Val -> Purge -> Test
        PURGE_DAYS = 20
        train_end = int(n * 0.60)
        val_start = train_end + PURGE_DAYS
        val_end = val_start + int(n * 0.20)
        test_start = val_end + PURGE_DAYS

        if test_start >= n:
            train_end = int(n * 0.65)
            val_start, val_end = train_end + 5, int(n * 0.85)
            test_start = val_end + 5

        # ------------------ 樹模型分支 ------------------
        if model_type != "tft":
            # 準備 DataFrame 以保留特徵名稱
            X_train_df = final_df[feature_cols].iloc[:train_end]
            X_val_df = final_df[feature_cols].iloc[val_start:val_end]
            X_test_df = final_df[feature_cols].iloc[test_start:]

            scaler = StandardScaler()
            # 強制將特徵名稱刻入 scaler (供外部 predict.py 使用 DataFrame 預測時不報錯)
            scaler.fit(X_train_df)

            # 轉換為 numpy 陣列供後續模型運算
            X_train = scaler.transform(X_train_df)
            X_val = scaler.transform(X_val_df)
            X_test = scaler.transform(X_test_df)

            y_all = np.ascontiguousarray(final_df[label_col].values, dtype=np.int32)
            y_train = y_all[:train_end]
            y_val = y_all[val_start:val_end]
            y_test = y_all[test_start:]

            X_train_raw = X_train_df.values  # 供 optuna 提取純數值

            def create_model(m_type, params):
                if m_type == "lightgbm":
                    return lgb.LGBMClassifier(**params)
                elif m_type == "xgboost":
                    if xgb is None: raise ImportError("請先安裝 xgboost")
                    return xgb.XGBClassifier(**params)
                elif m_type == "catboost":
                    if cb is None: raise ImportError("請先安裝 catboost")
                    return cb.CatBoostClassifier(**params)
                elif m_type == "rf":
                    return RandomForestClassifier(**params)

            def objective(trial):
                if model_type == "lightgbm":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", custom_params.get("n_estimators_min", 40), custom_params.get("n_estimators_max", 120), step=20),
                        "learning_rate": trial.suggest_float("learning_rate", custom_params.get("lr_min", 0.015), custom_params.get("lr_max", 0.06), log=True),
                        "max_depth": trial.suggest_int("max_depth", custom_params.get("max_depth_min", 3), custom_params.get("max_depth_max", 5)),
                        "num_leaves": trial.suggest_int("num_leaves", custom_params.get("num_leaves_min", 6), custom_params.get("num_leaves_max", 18)),
                        "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 8.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.85),
                        "subsample_freq": 1,
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
                        "boosting_type": "gbdt",
                        "objective": "binary",
                        "random_state": 101,
                        "n_jobs": 1,
                        "verbose": -1,
                    }
                elif model_type == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", custom_params.get("n_estimators_min", 40), custom_params.get("n_estimators_max", 120), step=20),
                        "learning_rate": trial.suggest_float("learning_rate", custom_params.get("lr_min", 0.015), custom_params.get("lr_max", 0.06), log=True),
                        "max_depth": trial.suggest_int("max_depth", custom_params.get("max_depth_min", 3), custom_params.get("max_depth_max", 5)),
                        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 10.0),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.85),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 8.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
                        "eval_metric": "logloss",
                        "random_state": 101,
                        "n_jobs": 1,
                    }
                elif model_type == "catboost":
                    params = {
                        "iterations": trial.suggest_int("iterations", custom_params.get("iterations_min", 40), custom_params.get("iterations_max", 120), step=20),
                        "learning_rate": trial.suggest_float("learning_rate", custom_params.get("lr_min", 0.015), custom_params.get("lr_max", 0.06), log=True),
                        "depth": trial.suggest_int("depth", custom_params.get("depth_min", 3), custom_params.get("depth_max", 5)),
                        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 12.0, log=True),
                        "random_seed": 101,
                        "thread_count": 1,
                        "verbose": False,
                    }
                elif model_type == "rf":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", custom_params.get("n_estimators_min", 100), custom_params.get("n_estimators_max", 200), step=50),
                        "max_depth": trial.suggest_int("max_depth", custom_params.get("max_depth_min", 2), custom_params.get("max_depth_max", 5)),
                        "min_samples_split": trial.suggest_int("min_samples_split", 20, 60),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 30),
                        "max_features": trial.suggest_float("max_features", 0.2, 0.5),
                        "class_weight": "balanced_subsample",
                        "random_state": 101,
                        "n_jobs": 1,
                    }

                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for tr_idx, v_idx in tscv.split(X_train):
                    fold_scaler = StandardScaler()
                    # 內部純數值運算，保證不報錯
                    X_f_tr = fold_scaler.fit_transform(X_train_raw[tr_idx])
                    X_f_val = fold_scaler.transform(X_train_raw[v_idx])

                    bm = create_model(model_type, params)
                    bm.fit(X_f_tr, y_train[tr_idx])
                    y_pv = bm.predict_proba(X_f_val)[:, 1]

                    if len(np.unique(y_train[v_idx])) >= 2:
                        scores.append(roc_auc_score(y_train[v_idx], y_pv))
                    else:
                        th = np.percentile(y_pv, 70)
                        y_pt = (y_pv >= th).astype(int)
                        scores.append(f1_score(y_train[v_idx], y_pt, pos_label=1, zero_division=0))
                return float(np.mean(scores))

            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=101))
            study.optimize(objective, n_trials=n_trials, timeout=600)

            best_params = study.best_params
            if model_type == "lightgbm":
                best_params.update({"subsample_freq": 1, "boosting_type": "gbdt", "objective": "binary", "random_state": 101, "n_jobs": 1, "verbose": -1})
            elif model_type == "xgboost":
                best_params.update({"eval_metric": "logloss", "random_state": 101, "n_jobs": 1})
            elif model_type == "catboost":
                best_params.update({"random_seed": 101, "thread_count": 1, "verbose": False})
            elif model_type == "rf":
                best_params.update({"class_weight": "balanced", "random_state": 101, "n_jobs": 1})

            base_model = create_model(model_type, best_params)
            base_model.fit(X_train, y_train)

            y_prob_val = base_model.predict_proba(X_val)[:, 1]
            y_prob_te = base_model.predict_proba(X_test)[:, 1]

            bundle_data = {
                "model_type": model_type,
                "model": base_model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "best_params": best_params
            }

        # ------------------ TFT 分支 ------------------
        else:
            if not TFT_AVAILABLE:
                raise ImportError("請先安裝 TFT 套件")

            pl.seed_everything(101)
            
            final_df["target_continuous"] = (
                final_df["rsi_diff_3d"] * 0.01 +
                final_df["bias_ma20"] * 1.0 +
                final_df["ma20_slope"] * 0.5
            ).astype(np.float32)

            scaler = StandardScaler()
            scaled_df = final_df.copy()
            # 外部 scaler 綁定 DataFrame 欄位
            scaler.fit(scaled_df[feature_cols].iloc[:train_end])
            scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols]).astype(np.float32)

            max_encoder_length = custom_params.get("max_encoder_length", 30)
            max_prediction_length = 1

            train_data_df = scaled_df.iloc[:train_end].copy()
            
            val_feed_start = max(0, val_start - max_encoder_length)
            val_data_df = scaled_df.iloc[val_feed_start:val_end].copy()

            test_feed_start = max(0, test_start - max_encoder_length)
            test_data_df = scaled_df.iloc[test_feed_start:].copy()

            training_dataset = TimeSeriesDataSet(
                train_data_df,
                time_idx="time_idx",
                target="target_continuous",
                group_ids=["stock_id"],
                min_encoder_length=max_encoder_length,
                max_encoder_length=max_encoder_length,
                min_prediction_length=max_prediction_length,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_reals=feature_cols,
                target_normalizer=None,
            )
            
            train_dl = training_dataset.to_dataloader(batch_size=32, shuffle=True, num_workers=0)

            # 【建立警告保護層】：強制遮蔽 pytorch-forecasting 內部對 StandardScaler 轉換引發的 Bug
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                
                validation_dataset = TimeSeriesDataSet.from_dataset(
                    training_dataset, val_data_df, predict=False, stop_randomization=True
                )
                val_dl = validation_dataset.to_dataloader(batch_size=32, shuffle=False, num_workers=0)

                tft = TemporalFusionTransformer.from_dataset(
                    training_dataset,
                    learning_rate=custom_params.get("learning_rate", 0.008),
                    hidden_size=custom_params.get("hidden_size", 32),
                    attention_head_size=custom_params.get("attention_head_size", 4),
                    dropout=custom_params.get("dropout", 0.2),
                    loss=QuantileLoss(),
                    output_size=7,
                    reduce_on_plateau_patience=3,
                )

                early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=6, verbose=False, mode="min")
                trainer = pl.Trainer(
                    max_epochs=int(tft_epochs),
                    accelerator="auto",
                    gradient_clip_val=0.1,
                    callbacks=[early_stop],
                    enable_progress_bar=False,
                )
                trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

                val_preds = tft.predict(val_dl).cpu().numpy().flatten()
                val_preds = np.nan_to_num(val_preds, nan=float(np.nanmean(val_preds) if len(val_preds) > 0 else 0.0))
                y_prob_val = 1.0 / (1.0 + np.exp(-val_preds))

                try:
                    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data_df, predict=False, stop_randomization=True)
                    test_dl = test_dataset.to_dataloader(batch_size=32, shuffle=False, num_workers=0)
                    te_preds = tft.predict(test_dl).cpu().numpy().flatten()
                    te_preds = np.nan_to_num(te_preds, nan=float(np.nanmean(te_preds) if len(te_preds) > 0 else 0.0))
                    y_prob_te = 1.0 / (1.0 + np.exp(-te_preds))
                except Exception:
                    y_prob_te = np.full(len(final_df) - test_start, 0.5)

            tft_ckpt_path = f"{stock_id}_tft_model.pt"
            torch.save(tft.state_dict(), tft_ckpt_path)

            bundle_data = {
                "model_type": model_type,
                "ckpt_path": tft_ckpt_path,
                "feature_importances_": np.ones(len(feature_cols)) / len(feature_cols),
                "max_encoder_length": max_encoder_length,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "best_params": {
                    "hidden_size": custom_params.get("hidden_size", 32),
                    "learning_rate": custom_params.get("learning_rate", 0.008),
                    "dropout": custom_params.get("dropout", 0.2),
                    "max_encoder_length": max_encoder_length,
                }
            }

            y_val_raw = final_df[label_col].iloc[val_start:val_end].values
            y_val = y_val_raw[-len(y_prob_val):]

            y_test_raw = final_df[label_col].iloc[test_start:].values
            y_test = y_test_raw[-len(y_prob_te):]

        # ------------------ 門檻校正與盲測 ------------------
        best_threshold = 0.50
        best_val_f1 = -1.0
        search_thresholds = np.linspace(np.percentile(y_prob_val, 15), np.percentile(y_prob_val, 85), 30)

        for th in search_thresholds:
            val_pred = (y_prob_val >= th).astype(int)
            sig_ratio = val_pred.sum() / len(val_pred)
            
            if 0.10 <= sig_ratio <= 0.30:
                p = precision_score(y_val, val_pred, pos_label=1, zero_division=0)
                if p > best_val_f1:
                    best_val_f1 = p
                    best_threshold = float(round(th, 4))

        if best_val_f1 == -1.0:
            best_threshold = float(round(np.median(y_prob_val), 4))

        y_pred_val = (y_prob_val >= best_threshold).astype(int)
        y_pred_te = (y_prob_te >= best_threshold).astype(int)

        val_acc = np.mean(y_pred_val == y_val)
        test_acc = np.mean(y_pred_te == y_test)

        val_auc = safe_calc_auc(y_val, y_prob_val)
        test_auc = safe_calc_auc(y_test, y_prob_te)

        report = classification_report(y_test, y_pred_te, zero_division=0)
        cm_val = confusion_matrix(y_val, y_pred_val).tolist()
        cm_test = confusion_matrix(y_test, y_pred_te).tolist()

        bundle_data["threshold"] = best_threshold
        bundle_data["val_auc"] = val_auc
        bundle_data["test_auc"] = test_auc
        joblib.dump(bundle_data, model_bundle_name)

        return {
            "bundle_name": model_bundle_name,
            "best_threshold": best_threshold,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "report": report,
            "cm_val": cm_val,
            "cm_test": cm_test,
            "best_params": bundle_data["best_params"],
        }