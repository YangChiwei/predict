# -*- coding: utf-8 -*-
#建立統一的資料庫連線與讀寫 helper，支援使用 SQLAlchemy 連接池與 Pandas 快速存取
#資料庫連接模組 (db_manager.py)
#Add db_lock decorator

# -*- coding: utf-8 -*-
"""
【資料存取管理層 (Storage Adapter)】
支援 MariaDB 與 本機 CSV 檔案模式快速切換
1.零修改上層程式碼：build_features_01.py、train_model_02.py、daily_predict_03.py 與 dashboard.py 完全不需改寫，依然沿用原有的 SQL 呼叫語法。
2.啟用 MariaDB 時，Set USE_DATABASE = True，
3.CSV 檔案: ./csv_data/ 目錄下（
包含 stock_raw_data.csv、stock_features.csv 與 stock_predictions.csv）

【資料存取管理層 (Storage Adapter)】
支援 MariaDB 與 純 CSV 模式（完全不依賴 SQLite）
"""

import os
import configparser
import pandas as pd
from contextlib import contextmanager

# ==========================================
# ⚙️ 全域儲存模式開關
# True : 連線 MariaDB 資料庫
# False: 停用 SQL，完全使用純本機 CSV (儲存於 ./csv_data/ 資料夾)
# ==========================================
#USE_DATABASE = True
USE_DATABASE = False
CSV_DATA_DIR = "csv_data"

if not USE_DATABASE:
    os.makedirs(CSV_DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. 鎖定機制 (SQL Named Lock vs 純檔案鎖)
# ----------------------------------------------------------------------
@contextmanager
def db_lock(lock_name: str, timeout: int = 15):
    """
    排他鎖管理：
    - USE_DATABASE=True : 使用 MariaDB 的 GET_LOCK
    - USE_DATABASE=False: 使用本機 .lock 標記檔案
    """
    if USE_DATABASE:
        from sqlalchemy import text
        engine = get_engine()
        conn = engine.connect()
        try:
            result = conn.execute(
                text("SELECT GET_LOCK(:name, :timeout)"),
                {"name": lock_name, "timeout": timeout}
            ).scalar()
            if result != 1:
                raise TimeoutError(f"無法取得資料庫鎖 [{lock_name}]，已有其他程序正在執行！")
            yield
        finally:
            try:
                conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
            except Exception:
                pass
            finally:
                conn.close()
    else:
        lock_file = os.path.join(CSV_DATA_DIR, f"{lock_name}.lock")
        try:
            with open(lock_file, "w") as f:
                f.write(f"locked at {pd.Timestamp.now()}")
            yield
        finally:
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except Exception:
                    pass

# ----------------------------------------------------------------------
# 2. 純 CSV 模式專用的 Engine 模擬器 (完全不使用 SQLite)
# ----------------------------------------------------------------------
class PureCSVSimulator:
    """模擬 SQLAlchemy Engine 與 Connection，攔截 DELETE 與 to_sql 並轉為 CSV 操作"""
    def __init__(self):
        self.pending_delete_sid = None

    @contextmanager
    def begin(self):
        yield self

    def execute(self, statement, parameters=None):
        # 攔截 DELETE FROM table WHERE stock_id = :sid
        if parameters and "sid" in parameters:
            self.pending_delete_sid = str(parameters["sid"]).strip()

# ----------------------------------------------------------------------
# 3. 資料庫 Engine 取得
# ----------------------------------------------------------------------
def get_engine(config_path="db_config.ini"):
    if not USE_DATABASE:
        return PureCSVSimulator()

    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔 {config_path}，請確認設定檔位置。")
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    db_cfg = config["database"]

    user = db_cfg.get("user")
    password = quote_plus(db_cfg.get("password"))
    host = db_cfg.get("host")
    port = db_cfg.getint("port", 3306)
    database = db_cfg.get("database", "taiwanstock")
    charset = db_cfg.get("charset", "utf8mb4")

    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
    return create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)

# ----------------------------------------------------------------------
# 4. 儲存模組 (save_to_db -> 支援 SQL Table 或 純 CSV 檔案)
# ----------------------------------------------------------------------
def save_to_db(df: pd.DataFrame, table_name: str, if_exists="append"):
    """將 DataFrame 存入 MariaDB 或純 CSV 檔案"""
    if USE_DATABASE:
        engine = get_engine()
        with engine.begin() as conn:
            df.to_sql(table_name, con=conn, if_exists=if_exists, index=False)
    else:
        csv_path = os.path.join(CSV_DATA_DIR, f"{table_name}.csv")
        if not os.path.exists(csv_path) or if_exists == "replace":
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            old_df = pd.read_csv(csv_path, dtype={"stock_id": str})
            # 若為追加模式且含有 stock_id，先剔除相同 stock_id 資料避免重複
            if "stock_id" in df.columns and "stock_id" in old_df.columns:
                incoming_sids = df["stock_id"].astype(str).unique()
                old_df = old_df[~old_df["stock_id"].astype(str).isin(incoming_sids)]
            combined_df = pd.concat([old_df, df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 5. 讀取模組 (read_from_db -> 支援 SQL Query 或純 CSV 讀取)
# ----------------------------------------------------------------------
def read_from_db(query: str, params=None) -> pd.DataFrame:
    """
    從 MariaDB 或純 CSV 讀取數據：
    - USE_DATABASE=False 時，直接透過 Pandas 讀取 CSV 進行查詢過濾
    """
    if USE_DATABASE:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(query), con=conn, params=params)
    else:
        params = params or {}
        sid = str(params.get("sid", "")).strip()

        # 1. 模擬 dashboard.py 聯合查詢 (features + raw_data)
        if "FROM stock_features" in query and "JOIN stock_raw_data" in query:
            feat_path = os.path.join(CSV_DATA_DIR, "stock_features.csv")
            raw_path = os.path.join(CSV_DATA_DIR, "stock_raw_data.csv")
            
            if not os.path.exists(feat_path) or not os.path.exists(raw_path):
                return pd.DataFrame()
                
            df_feat = pd.read_csv(feat_path, dtype={"stock_id": str})
            df_raw = pd.read_csv(raw_path, dtype={"stock_id": str})
            
            if sid:
                df_feat = df_feat[df_feat["stock_id"] == sid]
                df_raw = df_raw[df_raw["stock_id"] == sid]
                
            merged = pd.merge(
                df_feat, 
                df_raw[["stock_id", "date", "open", "high", "low", "close", "volume"]], 
                on=["stock_id", "date"], 
                how="inner"
            )
            merged = merged.rename(columns={
                "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
            })
            return merged.sort_values("date").reset_index(drop=True)

        # 2. 模擬 stock_features 查詢
        elif "stock_features" in query:
            feat_path = os.path.join(CSV_DATA_DIR, "stock_features.csv")
            if not os.path.exists(feat_path):
                return pd.DataFrame()
            df = pd.read_csv(feat_path, dtype={"stock_id": str})
            if sid:
                df = df[df["stock_id"] == sid]
            if "label IS NOT NULL" in query and "label" in df.columns:
                df = df.dropna(subset=["label"])
            return df.sort_values("date").reset_index(drop=True)

        # 3. 模擬 stock_raw_data 查詢
        elif "stock_raw_data" in query:
            raw_path = os.path.join(CSV_DATA_DIR, "stock_raw_data.csv")
            if not os.path.exists(raw_path):
                return pd.DataFrame()
            df = pd.read_csv(raw_path, dtype={"stock_id": str})
            if sid:
                df = df[df["stock_id"] == sid]
            return df.sort_values("date").reset_index(drop=True)

        # 4. 模擬 stock_predictions 查詢
        elif "stock_predictions" in query:
            pred_path = os.path.join(CSV_DATA_DIR, "stock_predictions.csv")
            if not os.path.exists(pred_path):
                return pd.DataFrame()
            df = pd.read_csv(pred_path, dtype={"stock_id": str})
            if sid:
                df = df[df["stock_id"] == sid]
            return df.sort_values("date").reset_index(drop=True)

        return pd.DataFrame()