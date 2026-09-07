# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # 補上這行避免 macOS 攔截 fork

import streamlit as st

# 1. 建立分頁物件
page_dashboard = st.Page("dashboard.py", title="決策互動儀表板", icon="📈", default=True)
page_pipeline = st.Page("pipeline_page.py", title="量化流水線管理", icon="⚙️")
page_benchmark = st.Page("benchmark_page.py", title="多模型評比", icon="🏆")

# 2. 設定側邊欄的選單順序
#pg = st.navigation([page_dashboard, page_pipeline, page_benchmark])
pg = st.navigation([page_dashboard, page_pipeline])
# 3. 啟動渲染
pg.run()
