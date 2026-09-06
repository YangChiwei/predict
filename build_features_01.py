# -*- coding: utf-8 -*-
"""
【特徵工程模組 - 嚴格遵守無前瞻偏差規範】
"""

import numpy as np
import pandas as pd
import yfinance as yf
from FinMind.data import DataLoader
from db_manager import read_from_db, save_to_db, db_lock

def _extract_series(df_or_s):
    """安全解包 yfinance 回傳的 Series 或 MultiIndex DataFrame"""
    if isinstance(df_or_s, pd.DataFrame):
        return df_or_s.iloc[:, 0].copy()
    return df_or_s.copy()

def run_build_features(stock_id="2330", start_date="2018-01-01", token=""):
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    stock_id = str(stock_id).strip()
    lock_key = f"sync_{stock_id}"

    with db_lock(lock_key, timeout=10):
        dl = DataLoader()
        if token:
            dl.login_by_token(api_token=token)

        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
        revenue_df = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date, end_date=end_date)

        stock_yf = yf.download(f"{stock_id}.TW", start=start_date, end=end_date, auto_adjust=True, progress=False)
        if stock_yf.empty:
            raise ValueError(f"查無股票代號 {stock_id} 的行情資料")

        close = _extract_series(stock_yf["Close"])
        volume = _extract_series(stock_yf["Volume"])
        open_p = _extract_series(stock_yf["Open"])
        high_p = _extract_series(stock_yf["High"])
        low_p = _extract_series(stock_yf["Low"])

        sox_yf = _extract_series(yf.download("^SOX", start=start_date, end=end_date, progress=False)["Close"])
        fx_yf = _extract_series(yf.download("TWD=X", start=start_date, end=end_date, progress=False)["Close"])
        vix_yf = _extract_series(yf.download("^VIX", start=start_date, end=end_date, progress=False)["Close"])

        raw_stock_df = pd.DataFrame({
            "stock_id": stock_id,
            "date": stock_yf.index.strftime("%Y-%m-%d"),
            "open": open_p.values,
            "high": high_p.values,
            "low": low_p.values,
            "close": close.values,
            "volume": volume.values,
            "sox_close": sox_yf.reindex(stock_yf.index).ffill().bfill().values,
            "usd_twd": fx_yf.reindex(stock_yf.index).ffill().bfill().values,
            "vix_close": vix_yf.reindex(stock_yf.index).ffill().bfill().values,
        })

        save_to_db(raw_stock_df, "stock_raw_data")

        # 1. 籌碼特徵
        inst_filtered = inst_df[inst_df["name"].isin(["Foreign_Investor", "Investment_Trust"])].copy()
        inst_daily = (
            inst_filtered.groupby(["date", "name"])["buy"].sum()
            - inst_filtered.groupby(["date", "name"])["sell"].sum()
        ).unstack(fill_value=0)
        net_buy_total = inst_daily.get("Foreign_Investor", 0) + inst_daily.get("Investment_Trust", 0)
        inst_daily["inst_net_buy_20d"] = net_buy_total.rolling(20).sum()
        inst_daily["inst_accel"] = net_buy_total.rolling(5).mean() - net_buy_total.rolling(20).mean()
        inst_daily.index = pd.to_datetime(inst_daily.index)

        # 2. 融資特徵
        margin_df["date"] = pd.to_datetime(margin_df["date"])
        margin_df = margin_df.set_index("date").sort_index()
        margin_df["margin_mom"] = margin_df["MarginPurchaseTodayBalance"].pct_change(20, fill_method=None)

        # 3. 營收特徵
        revenue_df["date"] = pd.to_datetime(
            revenue_df["revenue_year"].astype(str) + "-" + revenue_df["revenue_month"].astype(str) + "-11"
        )
        revenue_df = revenue_df.set_index("date").sort_index()
        revenue_df["revenue_mom"] = revenue_df["revenue"].pct_change(1, fill_method=None)

        # 4. 技術指標計算
        tech_df = pd.DataFrame(index=stock_yf.index)
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        tech_df["ma20_slope"] = (ma20 - ma20.shift(5)) / (ma20.shift(5) + 1e-6)
        tech_df["ma60_slope"] = (ma60 - ma60.shift(5)) / (ma60.shift(5) + 1e-6)
        tech_df["bias_ma20"] = (close - ma20) / (ma20 + 1e-6)
        tech_df["bias_ma60"] = (close - ma60) / (ma60 + 1e-6)
        tech_df["vol_ratio_5_20"] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-6)
        tech_df["volatility_20d"] = close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_14 = 100 - (100 / (1 + (gain / (loss + 1e-6))))
        tech_df["rsi_diff_3d"] = rsi_14 - rsi_14.shift(3)

        gain_long = (delta.where(delta > 0, 0)).rolling(70).mean()
        loss_long = (-delta.where(delta < 0, 0)).rolling(70).mean()
        tech_df["rsi_monthly"] = 100 - (100 / (1 + (gain_long / (loss_long + 1e-6))))

        ema12 = close.ewm(span=60, adjust=False).mean()
        ema26 = close.ewm(span=130, adjust=False).mean()
        tech_df["macd_monthly"] = ema12 - ema26

        tech_df["sox_mom"] = sox_yf.pct_change(20, fill_method=None).reindex(tech_df.index).ffill()
        tech_df["rs_vs_sox"] = close.pct_change(20, fill_method=None) - tech_df["sox_mom"]
        fx_aligned = fx_yf.reindex(tech_df.index).ffill()
        vix_aligned = vix_yf.reindex(tech_df.index).ffill()
        tech_df["usd_twd_mom20"] = fx_aligned.pct_change(20, fill_method=None)
        vix_ma20 = vix_aligned.rolling(20).mean()
        tech_df["vix_ma20_bias"] = (vix_aligned - vix_ma20) / (vix_ma20 + 1e-6)
        
        """
        # 5. 標籤產生 (未來 20 天)
        TARGET_GAIN, STOP_LOSS = 0.03, -0.03
        labels = []
        n_rows = len(close)
        for i in range(n_rows):
            if i + 20 >= n_rows:
                labels.append(np.nan)
                continue
            c_p = close.iloc[i]
            max_r = (high_p.iloc[i + 1 : i + 21].max() - c_p) / c_p
            min_r = (low_p.iloc[i + 1 : i + 21].min() - c_p) / c_p
            labels.append(1 if (max_r >= TARGET_GAIN and min_r > STOP_LOSS) else 0)

        tech_df["label"] = labels
        """        
        # 5. 標籤產生 (逐日路徑依賴檢驗：先碰停利者為 1，先碰停損者為 0)
        # Triple Barrier Method
        TARGET_GAIN = 0.05    # 目標停利 +5%
        STOP_LOSS = -0.025    # 目標停損 -2.5%
        LOOKAHEAD_DAYS = 20   # 最長觀察窗口
        
        high_series = stock_yf["High"].squeeze()
        low_series = stock_yf["Low"].squeeze()
        close_series = stock_yf["Close"].squeeze()
        
        labels = []
        n_rows = len(close_series)
        
        for i in range(n_rows):
            if i + LOOKAHEAD_DAYS >= n_rows:
                labels.append(np.nan)
                continue
                
            entry_price = close_series.iloc[i]
            outcome = 0
            
            # 逐日向後走訪 20 天，比對先觸發停利還是停損
            for d in range(1, LOOKAHEAD_DAYS + 1):
                day_high = high_series.iloc[i + d]
                day_low = low_series.iloc[i + d]
                
                high_ret = (day_high - entry_price) / entry_price
                low_ret = (day_low - entry_price) / entry_price
                
                # 若當日先跌破停損點，立即出局 (記為 0)
                if low_ret <= STOP_LOSS:
                    outcome = 0
                    break
                # 若當日先達標停利點，且未跌破停損，記為成功買點 (記為 1)
                elif high_ret >= TARGET_GAIN:
                    outcome = 1
                    break
            
            labels.append(outcome)

        tech_df["label"] = labels
        
        ######
        final_df = tech_df.copy()
        final_df["inst_net_buy_20d"] = inst_daily["inst_net_buy_20d"].reindex(final_df.index).ffill()
        final_df["inst_accel"] = inst_daily["inst_accel"].reindex(final_df.index).ffill()
        final_df["margin_mom"] = margin_df["margin_mom"].reindex(final_df.index).ffill()
        final_df["revenue_mom"] = revenue_df["revenue_mom"].reindex(final_df.index).ffill()

        feature_cols = [
            "inst_net_buy_20d", "inst_accel", "margin_mom", "revenue_mom",
            "ma20_slope", "ma60_slope", "bias_ma20", "bias_ma60",
            "vol_ratio_5_20", "volatility_20d", "rsi_diff_3d", "rsi_monthly",
            "macd_monthly", "sox_mom", "rs_vs_sox", "usd_twd_mom20",
            "vix_ma20_bias"
        ]

        # 數值填充與保護
        final_df[feature_cols] = final_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        final_df[feature_cols] = final_df[feature_cols].ffill().bfill().fillna(0.0)

        # 關鍵：特徵欄位不能有空值；label 允許最新 20 天為 NaN（供 UI 儀表板做即時推論）
        clean_df = final_df.dropna(subset=feature_cols).reset_index()
        clean_df["stock_id"] = stock_id
        clean_df["date"] = clean_df["Date"].dt.strftime("%Y-%m-%d")
    
        save_cols = ["stock_id", "date"] + feature_cols + ["label"]
        out_df = clean_df[save_cols].copy()

        save_to_db(out_df, "stock_features")
        return f"stock_raw_data ({stock_id})", f"stock_features ({stock_id})", out_df