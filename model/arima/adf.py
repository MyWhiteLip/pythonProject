import pandas as pd
import statsmodels.tsa.stattools as sts

# adfuller检验时间序列平稳性
def tagADF(t):
    result = pd.DataFrame(index=["Test Statistic Value", "p-value", "Lags Used",
                                 "Number of Observations Used",
                                 "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                          columns=['value']
                          )
    result['value']['Test Statistic Value'] = t[0]
    result['value']['p-value'] = t[1]
    result['value']['Lags Used'] = t[2]
    result['value']['Number of Observations Used'] = t[3]
    result['value']['Critical Value(1%)'] = t[4]['1%']
    result['value']['Critical Value(5%)'] = t[4]['5%']
    result['value']['Critical Value(10%)'] = t[4]['10%']
    print('t is:', t)
    return result

def getscore(t):
    adf_data = sts.adfuller(t)
    tagADF(adf_data)

