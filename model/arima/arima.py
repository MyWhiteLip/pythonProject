import numpy as np
import predeal.noise as pn
import predeal.test as imputer
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import adf as adf

df = pd.read_excel("..\\..\\source\\xiaguan19.xlsx")
# 缺失值补全
df = imputer.imput(df)
# 异常值处理: 滚动平均值
df['height'] = pn.remove_noise(df, 'height')
#
df.index = df['Date']
sub = df['2019-01-01 08:00':'2019-01-10 01:00']['height']
subtest = df['2019-1-01 09:00':'2019-1-10 10:00']['height']
sub = sub.dropna()
# 获得平稳性得分
adf.getscore(sub)

# 做出图像
plt.figure(figsize=(12, 5), dpi=80, linewidth=10)
plt.plot(sub)
plt.title('water height')
plt.xlabel('Date', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.show()
# 差分
sub_diff1 = sub.diff(1).dropna()
adf.getscore(sub_diff1)
adf.getscore(sub_diff1.diff(1).dropna())

# 建立模型
ARMAModel = sm.tsa.ARIMA(sub, order=(3, 1, 3), freq='H').fit()  # order=(p,d,q)
plt.figure(figsize=(11, 6))
# 样本内预测
predicts = ARMAModel.predict()

# 模型评价指标 1：计算 score
print(ARMAModel.fittedvalues)
delta = ARMAModel.fittedvalues
score = 1 - delta.var() / sub.var()
print('score:\n', score)
# 模型评价指标 2：使用均方根误差（RMSE）来评估模型样本内拟合的好坏。
# 利用该准则进行判别时，需要剔除“非预测”数据的影响。
train_vs = sub[predicts.index]  # 过滤没有预测的记录
plt.figure(figsize=(11, 6))
sub.plot(label='Original')
predicts.plot(label='Predict')
plt.legend(loc='best')
plt.title('RMSE: %.4f' % np.sqrt(sum((predicts - train_vs) ** 2) / train_vs.size))
plt.show()
# 预测接下来的20个时间点
f = ARMAModel.forecast(10)
plt.figure(figsize=(10, 10))
plt.plot(f)
plt.plot(subtest)
plt.show()
print(f)
