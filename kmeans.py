import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from math import atan, pi, radians
from numpy import rad2deg
import csv

def best_k_means(slope_list):
    slope_slope_list= slope(slope_list)
    return slope_slope_list.index(min(slope_slope_list))+1

def slope(wcc):
    slope_list = []
    for i in range(0, len(wcc) - 1):
        tanx = abs((wcc[i + 1] - wcc[i]) / 10000)
        slope_list.append(tanx)

    return slope_list


dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, :].values
y = dataset.iloc[:, 3].values
print(x[0][0])
print("_____________________")
print(y)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    labels = kmeans.labels_
    print(labels)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
# todo fix show
# todo add labels
plt.savefig("a.png")
slopes=slope(wcss)
print(best_k_means(slopes))


