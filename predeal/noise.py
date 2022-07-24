import pandas as pd
from Tools.demo.sortvisu import interpolate
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def remove_noise(t, a):
    # 滚动平均值去除异常值
    return t[a].rolling(2).mean()
