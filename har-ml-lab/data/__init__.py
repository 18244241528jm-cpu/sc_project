# HAR 数据集的一些常量配置
# 把这些数字写在这里，以后如果要改（比如采样频率变了），只改这里就行

# 原始信号的参数
N_TIMESTEPS = 128      # 每个样本的时间步数 (2.56s * 50Hz)
N_CHANNELS = 9         # 传感器通道数
SAMPLING_RATE = 50     # 采样频率 (Hz)

# 官方预计算特征的维度
N_OFFICIAL_FEATURES = 561

# 我们的信号通道名称
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

# 我们的自定义特征列表 (每个通道算这些)
STAT_METRICS = [
    "mean", "std", "max", "min", "median", "iqr", "energy"
]

# 计算出的自定义特征总维度
N_CUSTOM_FEATURES = N_CHANNELS * len(STAT_METRICS)  # 9 * 7 = 63

