import numpy as np
from sklearn.preprocessing import MinMaxScaler

import predeal.noise as pn
import predeal.test as imputer
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import model.arima.adf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os
from keras.models import Sequential, load_model

df = pd.read_excel("..\\..\\source\\xiaguan19.xlsx",usecols=[0,1],skipfooter=3)
df=df.head(261)
# 缺失值补全
print(df)
df = imputer.imput(df)
# 异常值处理: 滚动平均值
df['height'] = pn.remove_noise(df, 'height')
df=df.dropna(axis=0,how='any')
dataset = df['height'].values
# 将整型变为float
dataset = dataset.astype('float32')
# 归一化 在下一步会讲解
train_size = int(len(dataset) * 0.65)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

def create_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 训练数据太少 look_back并不能过大
look_back = 1
trainX, trainY = create_dataset(trainlist, look_back)
testX, testY = create_dataset(testlist, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)
model.save(os.path.join("DATA", "Test" + ".h5"))

# model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 反归一化

plt.plot(trainY)
plt.plot(trainPredict[1:])
vp=trainPredict[1:]
vs = np.delete(trainY,[0,0])
sump=0
for i in range(len(vs)):
    sump+=(vp[i]-vs[i])**2
plt.title('RMSE: %.4f' % np.sqrt((sump) / vs.size))
plt.show()
