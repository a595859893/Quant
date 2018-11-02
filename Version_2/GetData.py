import os
import pickle
import hashlib
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import AutoDateLocator
from keras.layers import GRU, Dense, Dropout, Masking, Input, Flatten, TimeDistributed
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


def fomatData(stock: dict, indicator: list, length: int, further: int):
    assert(len(stock) == batch_size)

    # 变量初始化
    data = np.zeros(shape=(len(stock), length, 4), dtype=np.float)
    label = np.zeros(shape=(len(stock), length, 4*further), dtype=np.float)

    # 数据规范处理

    counter = 0

    for key, df in stock.items():
        df.fillna(0)  # 筛除不合法数据

        scaler = {key: df.at[0, key] for _, key in enumerate(indicator)}
        max_len = df.shape[0]-further
        previous = {}
        for index, row in df.iterrows():
            for subindex, key in enumerate(indicator):
                for i in range(further):
                    if(index > i and index-i < max_len):
                        label[counter, index-i-1,
                              subindex+i*len(indicator)] = row[key]/scaler[key]
                        # label[counter, index-1,
                        #       subindex] = 1 if(previous[key] < row[key]) else 0

                if(index < max_len):
                    data[counter, index, subindex] = row[key]/scaler[key]
                    previous[key] = row[key]
        counter += 1
    return data, label


def showData(predicts: np.ndarray, data: np.ndarray, indicator: list, stocks: dict, further: int):
    counter = -1
    colorList = ['r', 'g', 'b', 'k', 'm', 'c', '#66ccff', '#ffcc00']
    for _, stock in stocks.items():
        # 选择股票
        counter += 1
        scaler = {key: stock.at[0, key] for _, key in enumerate(indicator)}
        data_time = [datetime.strptime(d, "%Y%m%d").date()
                     for d in stock["trade_date"]]
        predicted = predicts[counter]
        dataRow = data[counter]
        length = stock.shape[0] - further
        # 选择子图表
        plt.figure(counter)

        # 配置时间坐标轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())  # 时间间隔自动选取

        # 绘制历史数据
        plt.plot(data_time[:length], stock["close"].values[:length],
                 label="stock", c="blue", lw=0.5)
        # plt.twinx()
        close_index = 1+len(indicator)*(further-1)
        print(dataRow[:length, close_index])
        plt.plot(data_time[further:length+further], dataRow[:length, close_index].T*scaler['close'],
                 label="data", c="green", lw=0.5)
        plt.plot(data_time[further:length+further], predicted[:length, close_index].T*scaler['close'],
                 label="pred", c="red", lw=0.5)
        # plt.scatter(data_time[:length], dataRow[:length, 1],
        #             label="data_low", c='b', s=2)
        # plt.scatter(data_time[:length], predicted[:length, 1],
        #             label="pred_low", c='g', s=2)
        # wrong = np.sign(predicted-0.5) == dataRow*2-1
        # wrong = wrong*0.5+0.25
        # plt.scatter(data_time[:length], wrong[:length, 1].T,
        #             label="wrong", c='r', s=5)

        # plt.twinx()
        # for index, key in enumerate(indicator):
        #     plt.scatter(data_time[:length],
        #                 dataRow[:length, index],
        #                 label="data_%s" % key,
        #                 c=colorList[index], alpha=0.2,
        #                 s=2)
        #     plt.scatter(data_time[:length],
        #                 predicted[:length, index],
        #                 label="pred_%s" % key,
        #                 c=colorList[index], alpha=0.4,
        #                 s=2)
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

        # 绘图细节
        plt.grid(True)
        plt.axis("tight")
        plt.xlabel("Time", size=20)
        plt.ylabel("Value", size=20)
        plt.title("Graph", size=20)
        plt.legend(loc=0)  # 添加图例

    plt.show()  # 显示画布


# 常数设定
epochs = 50
max_length = 250
further_step = 20
indicator = ["open", "close", "low", "vol"]
codelist = ["600094.SH", "603818.SH", "600507.SH",
            "601377.SH", "600309.SH", "600298.SH", "600018.SH"]

batch_size = len(codelist)

# 数据准备
hs = get_hs("20160101", "20161201", codelist)
data, label = fomatData(hs, indicator, max_length, further_step)

# 模型设置
model = Sequential()
model.add(Masking(0, input_shape=(max_length, 4)))
model.add(GRU(units=64, input_shape=(None, 4),
              return_sequences=True, implementation=2))
model.add(Dropout(0.2))
model.add(GRU(units=32, return_sequences=True, implementation=2))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(64)))
model.add(Dropout(0.2))
model.add(Dense(4*further_step))
model.compile(loss='mse',
              optimizer='rmsprop', metrics=['accuracy'])

# 模型训练与预测
model.fit(data, label, epochs=epochs, batch_size=batch_size)


testcodelist = ["600252.SH", "600315.SH",
                "600383.SH", "600299.SH", "600026.SH", "603993.SH"]
batch_size = len(testcodelist)
hs = get_hs("20160101", "20161101", testcodelist)
data, label = fomatData(hs, indicator, max_length, further_step)
model.reset_states()
predicted = model.predict(data, batch_size=batch_size)

# 显示结果
showData(predicted, label, indicator, hs, further_step)
