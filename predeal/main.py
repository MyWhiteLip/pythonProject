import pandas as pd
from Tools.demo.sortvisu import interpolate
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import  noise
df=pd.read_excel("..\\source\\xiaguan19.xlsx")
df=df.head(300)
df['rolling']=noise.remove_noise(df,'height')
figure(figsize=(12, 5), dpi=80, linewidth=10)
plt.plot(df['Date'],df['height'])
plt.plot(df['Date'],df['rolling'])
plt.legend(["origin",'rolling mean'])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Height', fontsize=14)
plt.show()

# df['Linear'] = df['height'].interpolate(method='linear')
# df['Spline order 3'] = df['height'].interpolate(method='spline', order=4)
# df['quadratic']=df['height'].interpolate(method='quadratic')
# df['akima']=df['height'].interpolate(method='akima')
# methods = ['Linear', 'Spline order 3','quadratic','akima']
# # 缺失值补全方法（存在问题）
# for method in methods:
#     figure(figsize=(12, 5), dpi=80, linewidth=10)
#     plt.plot(df['Date'], df[method])
#     plt.title('River height of xiaguan'+' '+ method)
#     plt.xlabel('Date', fontsize=14)
#     plt.ylabel('Height', fontsize=14)
#     plt.show()

