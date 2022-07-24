import pandas as pd
from Tools.demo.sortvisu import interpolate
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import  noise as rn
dt = pd.read_excel("..\\source\\test.xlsx")
dt=dt.head(250)
dt['Date'] = pd.to_datetime(dt['Date'])

dt.sort_values(by=['Date'], inplace=True, ascending=True)
print(dt)
#原始表格
figure(figsize=(12, 5), dpi=80, linewidth=10)
plt.plot(dt['Date'], dt['height'])
plt.title('water height')
plt.xlabel('Date', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.show()
