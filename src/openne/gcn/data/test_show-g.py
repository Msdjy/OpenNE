import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           #加载PCA算法包
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np

x, y= [], []
with open("labels.txt", "r") as f:  # 打开文件
    data1 = f.read().split("\n")  # 读取文件
    for i in data1:
        y.append(int(i))
with open("embeddings.txt", "r") as f:  # 打开文件
    data1 = f.read().split("\n")  # 读取文件
    for item in data1:
        a = []
        item1 = item.split(" ")
        for i in item1:
            a.append(float(i))
        x.append(a)

# pca=PCA(n_components=3)     #加载PCA算法，设置降维后主成分数目为2
# embedded=pca.fit_transform(x)#对样本进行降维

tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(x)

print(embedded)

# #可视化
color = ['#F0F8FF', 'green', 'b', 'r', '#7FFFD4', '#FFC0CB', '#00022e']

# 创建显示的figure
fig = plt.figure()
ax = plt
# ax = Axes3D(fig)

print(type(y[:]))
print(type(embedded[:, 0]))
ax.scatter(embedded[:, 0], embedded[:, 1], c=[color[t] for t in y], alpha=0.5, s=np.random.randint(5, 10))

#生成三维数据
# xx = np.random.random(20)*10-5   #取100个随机数，范围在5~5之间
# yy = np.random.random(20)*10-5
# X, Y = np.meshgrid(xx, yy)
# Z = np.sin(np.sqrt(X**2+Y**2))
# ax.scatter(X, Y, Z, c=np.random.random(400), alpha=0.5, s=np.random.randint(5, 10))


# 关闭了plot的坐标显示
# plt.axis('off')
plt.show()

