import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. 加载并清洗数据集
def load_and_clean_data(dataset_path, stock_symbol="AAPL"):
    """
    加载指定股票的数据集并清洗
    :param dataset_path: kaggle数据集下载路径
    :param stock_symbol: 股票代码（默认苹果AAPL）
    :return: 清洗后的DataFrame（仅保留日期、收盘价）
    """
    # 遍历数据集文件夹，找到目标股票的CSV文件（数据集按字母分文件夹）
    import os
    stock_file = None
    # 数据集文件结构：按字母分文件夹，比如AAPL在Stocks/a开头的文件夹
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower() == f"{stock_symbol.lower()}.us.txt":
                stock_file = os.path.join(root, file)
                break
        if stock_file:
            break
    
    if not stock_file:
        raise ValueError(f"未找到{stock_symbol}的股票数据，请检查代码")
    
    # 加载数据并筛选列
    df = pd.read_csv(
        stock_file,
        usecols=["Date", "Close"],  # 仅保留日期和收盘价（核心预测目标）
        parse_dates=["Date"],       # 解析日期格式
        index_col="Date"            # 日期设为索引
    )
    
    # 清洗：删除缺失值、重复值
    df = df.dropna().drop_duplicates()
    # 按日期排序
    df = df.sort_index()
    
    return df

# 2. 构建时序数据（将序列数据转换为监督学习格式）
def create_sequences(data, seq_length=60):
    """
    将时序数据转换为LSTM输入格式（X: 前seq_length天数据，y: 第seq_length+1天收盘价）
    :param data: 归一化后的收盘价数组
    :param seq_length: 时间步长（默认用前60天预测下1天）
    :return: X(特征), y(标签)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])  # 前60天数据作为特征
        y.append(data[i, 0])              # 第61天数据作为标签
    # 转换为numpy数组（适配LSTM输入）
    X = np.array(X)
    y = np.array(y)
    # 调整X形状：(样本数, 时间步长, 特征数) → LSTM要求3维输入
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

# 3. 计算评估指标（MAE、MSE、RMSE，满足作业至少3项指标要求）
def calculate_metrics(y_true, y_pred):
    """
    计算回归任务的评估指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # 计算MAPE（避免分母为0，加小常数）
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2)  # 百分比，保留2位小数
    }

# 4. 可视化预测结果
def plot_predictions(y_true, y_pred, save_path="prediction_plot.png"):
    """
    绘制真实值vs预测值曲线
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="真实收盘价", color="blue")
    plt.plot(y_pred, label="预测收盘价", color="red", alpha=0.7)
    plt.title("美股收盘价预测结果（LSTM模型）")
    plt.xlabel("时间步")
    plt.ylabel("归一化收盘价")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()