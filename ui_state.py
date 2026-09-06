import os
import json

UI_CONFIG_FILE = "ui_config.json"

def get_ui_stock():
    """讀取 UI 最後一次輸入的股票代號，預設 2330"""
    if os.path.exists(UI_CONFIG_FILE):
        try:
            with open(UI_CONFIG_FILE, "r") as f:
                return json.load(f).get("stock_id", "2330")
        except Exception:
            pass
    return "2330"

def set_ui_stock(stock_id):
    """將 UI 新輸入的代號存檔"""
    with open(UI_CONFIG_FILE, "w") as f:
        json.dump({"stock_id": str(stock_id).strip()}, f)