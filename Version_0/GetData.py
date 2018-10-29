import os
import pickle
import hashlib
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import AutoDateLocator
from keras.layers import GRU, Dense
from keras.models import Sequential
from datetime import datetime
from pandas import DataFrame

pro = ts.pro_api(
    "ae9f4f65551123476a01659ecfd124dd832881d99b57574f64f8d333")


# 获取沪股通、深股通成分
def get_hs(start: str, end: str, codeList: list = None, ktype="D"):
    stocks = {}

    if (codeList is not None):
        info_str = hashlib.md5("".join(codeList).encode('utf-8')).hexdigest()
        path = info_str+start+'-'+end+ktype + ".pkl"
        # 若本地存在所需数据
        if os.path.exists(path):
            print("file exits, loading...")
            # 读取数据
            with open(path, "rb") as file:
                stocks = pickle.load(file)
            return stocks
    else:
        codeList = pro.hs_const(hs_type='SH')["ts_code"]

    for code in codeList:
        # 获取指定代号的历史数据
        print("download %s" % code)
        stocks[code] = pro.daily(
            ts_code=code, start_date=start, end_date=end)
    # 存储数据
    with open(path, "wb") as file:
        pickle.dump(stocks, file)

    print("finish")

    return stocks


def fomatData(df: DataFrame):
    # 构建数组格式
    length = df.shape[0]
    train_data = np.ndarray(shape=(length, 1, 3))
    train_label = np.ndarray(shape=(length, 3))

    # 缩放因子
    scaleFactor = {}
    scaleFactor["open"] = df["open"][0]
    scaleFactor["close"] = df["close"][0]
    scaleFactor["low"] = df["low"][0]

    # 数据规范处理
    data = df.loc[:, ['open', 'close', 'low']]
    for index, row in data.iterrows():
        if(index > 0):
            train_data[index-1, 0, 0] = row['open']/scaleFactor["open"]
            train_data[index-1, 0, 1] = row['close']/scaleFactor["close"]
            train_data[index-1, 0, 2] = row['low']/scaleFactor["low"]

        if(index < length-1):
            train_label[index, 0] = row['open']/scaleFactor["open"]
            train_label[index, 1] = row['close']/scaleFactor["close"]
            train_label[index, 2] = row['low']/scaleFactor["low"]

    return train_data, train_label, length


def showData(stock, predict):
    data_time = [datetime.strptime(d, "%Y%m%d").date()
                 for d in stock["trade_date"]]

    # 配置时间坐标轴
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(AutoDateLocator())  # 时间间隔自动选取

    # 绘制历史数据
    plt.plot(data_time, stock["low"].values,
             label="low", color="yellow", lw=1.5)
    plt.plot(data_time, predicted[:, 0]*stock["low"][0],
             label="predicted_low", color="red", lw=1.5)

    plt.gcf().autofmt_xdate()  # 自动旋转日期标记

    # 绘图细节
    plt.grid(True)
    plt.axis("tight")
    plt.xlabel("Time", size=20)
    plt.ylabel("Price", size=20)
    plt.title("Graph", size=20)
    plt.legend(loc=0)  # 添加图例

    plt.show()  # 显示画布


# 常数设定
batch_size = 1
epochs = 10
codelist = ["600094.SH"]

# 数据准备
hs = get_hs("20160101", "20161201", codelist)
stock = hs[codelist[0]]
data, label, length = fomatData(stock)  # 选取603818.SH历史数据，并规范化

# 模型设置
model = Sequential()
model.add(GRU(units=16, input_shape=(1, 3),
              stateful=True, batch_size=batch_size))
model.add(Dense(3))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# 模型训练与预测
model.fit(data, label, epochs=epochs, batch_size=batch_size)
model.reset_states()
predicted = model.predict(data, batch_size=batch_size)

# 显示结果
showData(stock, predicted)
