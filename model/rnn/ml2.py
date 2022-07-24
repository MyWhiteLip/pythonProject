import datetime

import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import predeal.noise as pn
import predeal.test as imputer


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def pre(model, x_input):
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]


# define input sequence
df = pd.read_excel("..\\..\\source\\xiaguan19.xlsx", usecols=[0, 1], skipfooter=3)
df = imputer.imput(df)
df['height'] = pn.remove_noise(df, 'height')
df = df.dropna(axis=0, how='any')
df.index = df['Date']
df_train = df['2019-01-10 08:00':'2019-5-20 01:00']['height'].values
df_check = df['2019-05-20 02:00':'2019-5-20 05:00']['height']
df_real = df['2019-01-10 09:00':'2019-5-20 05:00']['height']
comp = df_check
dataset = df_train
print(dataset.size)
datasize = dataset.size
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(dataset, n_steps)
print(X, y)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))  # 隐藏层，输入，特征维
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, batch_size=1, verbose=2)  # 迭代次数，批次数，verbose决定是否显示每次迭代
# demonstrate prediction
x_input = array([dataset[datasize - 3], dataset[datasize - 2], dataset[datasize - 1]])
x_input = x_input.reshape((1, n_steps, n_features))
ans_list = []
for i in range(4):
    ans = pre(model, x_input)
    ans_list.append(ans)
    x_input[0][0] = x_input[0][1]
    x_input[0][1] = x_input[0][2]
    x_input[0][2] = ans

x = model.predict(X)
list=[]
for i in range(4):
    list.append(datetime.datetime(2019,1,20,i+2,0))
predict=pd.Series(data=ans_list,index=list)
plt.figure(figsize=(10, 10))
plt.legend(loc='best')
plt.plot(predict)
plt.plot(df_check)
plt.legend(['pre', 'real'])
plt.show()
