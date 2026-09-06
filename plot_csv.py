import pandas as pd
import plotly.express as px

# 讀取評比產出的 CSV
df = pd.read_csv("benchmark_results_2330_20260830_0833.csv")

# 1. 繪製買進勝率排行榜長條圖
fig_bar = px.bar(
    df, 
    x="Model", 
    y="Test_Precision_BUY", 
    color="Test_Precision_BUY",
    text_auto=".2%",
    title="各模型盲測集「買進勝率 (Precision)」排行榜",
    color_continuous_scale="Viridis"
)
fig_bar.show()

# 2. 繪製過擬合 vs 勝率 象限圖
fig_scatter = px.scatter(
    df,
    x="Overfit_Gap",
    y="Test_Precision_BUY",
    text="Model",
    size="Test_Signals_Total",
    color="Test_Accuracy",
    title="模型穩健度評估 (左上方為最穩健且高勝率)",
    labels={"Overfit_Gap": "過擬合差距 (Val Acc - Test Acc)", "Test_Precision_BUY": "買進勝率"}
)
fig_scatter.show()