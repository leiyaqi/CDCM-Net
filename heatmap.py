import torch
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

index = range(2, 4)
dataset = np.array(pd.read_csv('data_chat.csv', usecols=index))
dataset = MaxMinNormalization(dataset)
dataset1 = dataset[:, 0]
dataset2 =dataset [:, 1]
shp = int(np.sqrt(len(dataset)))
dataset1 = dataset1[:shp*shp]
dataset1 = dataset1.reshape(shp,shp)
dataset2 = dataset2[:shp*shp]
dataset2 = np.abs(dataset2-1.0)
dataset2 = dataset2.reshape(shp,shp)
# print(dataset)
# print(dataset)
plt.figure(figsize=(14, 6))
# ax1 = sns.heatmap(dataset1, cmap="YlGnBu")
plt.subplot(121)
plt.title("Img Feature Confidence Score")
ax1 = sns.heatmap(dataset1, cmap="YlGnBu")
plt.subplot(122)
plt.title("Text Feature Confidence Score")
ax2 = sns.heatmap(dataset2, cmap="YlGnBu")

plt.show()
