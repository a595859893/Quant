import os
import pickle
import hashlib
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import AutoDateLocator
from keras.layers import GRU, Dense, Dropout
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


def fomatData(stock: dict, batch_size: int, length: int):
    assert(len(stock) == batch_size)

    length = -1
    for key in stock:
        # 获取最小长度并初始化
        if(length == -1):
            length = stock[key].shape[0]
        else:
            length = min(length, stock[key].shape[0])

    length -= 1
    train_data = np.zeros(shape=(length*batch_size, 1, 4), dtype=np.float)
    train_label = np.zeros(shape=(length*batch_size, 4), dtype=np.float)

    counter = -1
    for key in stock:
        # 获取并处理DataFrame
        df = stock[key]
        df.fillna(0.01)

        counter += 1
        # 缩放因子
        scaleFactor = {}
        scaleFactor["open"] = df["open"][0]
        scaleFactor["close"] = df["close"][0]
        scaleFactor["low"] = df["low"][0]
        scaleFactor["vol"] = df["vol"][0]

        # 数据规范处理
        data = df.loc[:, ['open', 'close', 'low', 'vol']]
        for index, row in data.iterrows():
            last = (index-1) * batch_size + counter
            current = index * batch_size + counter

            if(index > 0):
                train_label[last, 0] = row['open'] / scaleFactor["open"]
                train_label[last, 1] = row['close'] / scaleFactor["close"]
                train_label[last, 2] = row['low'] / scaleFactor["low"]
                train_label[last, 3] = row['vol'] / scaleFactor["vol"]

            if(index >= length):
                break

            train_data[current, 0, 0] = row['open'] / scaleFactor["open"]
            train_data[current, 0, 1] = row['close'] / scaleFactor["close"]
            train_data[current, 0, 2] = row['low'] / scaleFactor["low"]
            train_data[current, 0, 2] = row['vol'] / scaleFactor["vol"]

    return train_data,  train_label, length


def showData(stocks, predicts, data, length, batch_size):
    counter = 0
    for key in stocks:
        # 选择股票
        counter += 1
        stock = stocks[key]
        frameList = [i*batch_size+counter-1 for i in range(length)]
        predicted = predicts[frameList]
        dataRow = data[frameList]

        # 选择子图表
        plt.figure(counter)

        data_time = [datetime.strptime(d, "%Y%m%d").date()
                     for d in stock["trade_date"]]
        # 配置时间坐标轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())  # 时间间隔自动选取

        # 绘制历史数据
        plt.plot(data_time[1:length + 1], dataRow[:length, 1]*stock["close"][0],
                 label="data_low", color="yellow", lw=1.5)
        plt.plot(data_time[:], stock["close"].values[:],
                 label="low", color="blue", lw=1.5)
        plt.plot(data_time[1:length + 1], predicted[:length, 1]*stock["close"][0],
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
epochs = 100
codelist = ["600094.SH", "603818.SH", "600507.SH",
            "601377.SH", "600309.SH", "600298.SH", "600018.SH"]

batch_size = len(codelist)

# 数据准备
hs = get_hs("20160101", "20161201", codelist)
data, label, length = fomatData(hs, batch_size, hs)

# 模型设置
model = Sequential()
model.add(GRU(units=64, input_shape=(1, 4),
              stateful=True, batch_size=batch_size, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=32, stateful=True,
              batch_size=batch_size, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=16,  stateful=True, batch_size=batch_size))
model.add(Dropout(0.2))
model.add(Dense(8, activation="tanh"))
model.add(Dense(4))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

# 模型训练与预测
model.fit(data, label, epochs=epochs, batch_size=batch_size, shuffle=False)


testcodelist = ["600252.SH", "600291.SH", "600315.SH",
                "600383.SH", "600299.SH", "600026.SH", "603993.SH"]
batch_size = len(testcodelist)
hs = get_hs("20160101", "20161201", testcodelist)
data, label, length = fomatData(hs, batch_size, hs)
model.reset_states()
predicted = model.predict(data, batch_size=batch_size)

# 显示结果
showData(hs, predicted, label, length, batch_size)
