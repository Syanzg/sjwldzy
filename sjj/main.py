import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import kagglehub
import utils  # 导入自定义工具函数

# ====================== 1. 配置参数 ======================
STOCK_SYMBOL = "AAPL"          # 目标股票（苹果AAPL，可替换为GOOG、MSFT等）
SEQ_LENGTH = 60                # 时间步长：用前60天预测下1天
TRAIN_SIZE = 0.8               # 训练集比例（80%训练，20%测试）
EPOCHS = 50                    # 训练轮数
BATCH_SIZE = 32                # 批次大小
LEARNING_RATE = 0.001          # 学习率

# ====================== 2. 加载并预处理数据 ======================
print("===== 加载数据集 =====")
# 下载/获取数据集路径（已下载过则直接读取）
dataset_path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
print(f"数据集路径：{dataset_path}")

# 加载并清洗数据
df = utils.load_and_clean_data(dataset_path, STOCK_SYMBOL)
print(f"\n{STOCK_SYMBOL}数据预览：")
print(df.head())
print(f"\n数据时间范围：{df.index.min()} 至 {df.index.max()}")
print(f"数据总行数：{len(df)}")

# 归一化（LSTM对数值敏感，需缩放到0-1之间）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# 构建时序序列
X, y = utils.create_sequences(scaled_data, SEQ_LENGTH)
print(f"\n序列数据形状：X={X.shape}, y={y.shape}")

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_SIZE, shuffle=False  # 时序数据不能打乱！
)
print(f"训练集：X={X_train.shape}, y={y_train.shape}")
print(f"测试集：X={X_test.shape}, y={y_test.shape}")

# ====================== 3. 构建LSTM模型 ======================
print("\n===== 构建LSTM模型 =====")
model = Sequential([
    # 第一层LSTM，返回序列（供下一层LSTM使用）
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),  # 防止过拟合
    
    # 第二层LSTM，不返回序列
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # 全连接层
    Dense(units=25),
    Dense(units=1)  # 输出层：预测收盘价（单值）
])

# 编译模型
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mean_squared_error"  # 回归任务用MSE损失
)

# 打印模型结构
model.summary()

# ====================== 4. 训练模型 ======================
print("\n===== 开始训练模型 =====")
# 早停策略：验证集损失不再下降时停止，避免过拟合
early_stop = EarlyStopping(
    monitor="val_loss",  # 监控验证集损失
    patience=5,          # 5轮无改善则停止
    restore_best_weights=True  # 恢复最优权重
)

# 训练
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ====================== 5. 模型评估 ======================
print("\n===== 模型评估 =====")
# 预测
y_pred = model.predict(X_test)
# 反归一化（恢复真实价格）
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

# 计算评估指标
metrics = utils.calculate_metrics(y_test_original, y_pred_original)
print("评估指标：")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# ====================== 6. 结果可视化 ======================
print("\n===== 绘制预测结果 =====")
utils.plot_predictions(y_test_original, y_pred_original)

# ====================== 7. 保存模型 ======================
model.save("lstm_stock_prediction_model.h5")
print("\n模型已保存为：lstm_stock_prediction_model.h5")
print("===== 实验完成 =====")