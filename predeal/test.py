import pandas as pd
def imput(t):
    t['Date'] = pd.to_datetime(t['Date'])
    helper = pd.DataFrame({'Date': pd.date_range(t['Date'].min(), t['Date'].max(), freq='H')})
    df = pd.merge(t, helper, on='Date', how='outer').sort_values('Date')
    #线性补全
    df['height'] = df['height'].interpolate(method='linear')
    return df