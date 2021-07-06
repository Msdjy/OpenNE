import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           #加载PCA算法包
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np
import os
import imageio

# from PIL import Image
# from images2gif import writeGif



# 可视化
def write_vec(outs, labels, epoch) :
    label_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}  # 定义标签颜色字典
    # 写文件
    with open("./embeddings.txt", "w") as fe, open("./labels.txt", 'w') as fl:
        for i in range(len(outs[3])):
            fl.write(label_dict[int(list(labels[i]).index(1.))] + "\n")
            fe.write(" ".join(map(str, outs[3][i])) + "\n")
    # print(os.getcwd())
    with open(os.path.join('openne','gcn','tmp', str(epoch) + "embeddings.txt"), "w") as fe:
        for i in range(len(outs[3])):
            fe.write(" ".join(map(str, outs[3][i])) + "\n")

# gif显示问题
# def img2gif(epoch):
#     outfilename = "my.gif"  # 转化的GIF图片名称
#     filenames = []  # 存储所需要读取的图片名称
#     for i in range(epoch):  # 读取100张图片
#         # (os.path.join('tmp', t + ".jpg"))
#         filename = (os.path.join('tmp', str(i) + ".jpg"))  # path是图片所在文件，最后filename的名字必须是存在的图片
#         filenames.append(filename)  # 将使用的读取图片汇总
#     frames = []
#     for image_name in filenames:  # 索引各自目录
#         im = Image.open(image_name)  # 将图片打开，本文图片读取的结果是RGBA格式，如果直接读取的RGB则不需要下面那一步
#         im = im.convert("RGB")  # 通过convert将RGBA格式转化为RGB格式，以便后续处理
#         im = np.array(im)  # im还不是数组格式，通过此方法将im转化为数组
#         frames.append(im)  # 批量化
#     writeGif(outfilename, frames, duration=0.1, subRectangles=False)  # 生成GIF，其中durantion是延迟，这里是1ms

def img2gif(epoch):
    img_paths = []
    for i in range(epoch):  # 读取100张图片
        # (os.path.join('tmp', t + ".jpg"))
        # img_path = (os.path.join('tmp', str(i) + ".jpg"))  # path是图片所在文件，最后filename的名字必须是存在的图片
        img_path = (os.path.join('openne', 'gcn', 'tmp', str(i) + ".jpg"))  # path是图片所在文件，最后filename的名字必须是存在的图片
        img_paths.append(img_path)  # 将使用的读取图片汇总

    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("test.gif", gif_images, fps=10)



def vec2img(epoch, G):
    x, y = [], []
    with open("labels.txt", "r") as f:  # 打开文件
        data1 = f.read().split("\n")  # 读取文件
        t = 1
        for i in data1:
            t = t + 1
            if t == 500:
                break
            y.append(int(i))
    with open("embeddings.txt", "r") as f:  # 打开文件
        data1 = f.read().split("\n")  # 读取文件
        t = 1
        for item in data1:
            t = t + 1
            if t == 500:
                break
            a = []
            item1 = item.split(" ")
            for i in item1:
                a.append(float(i))
            x.append(a)


    pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
    embedded=pca.fit_transform(x)#对样本进行降维

    # tsne = TSNE(n_components=2)
    # embedded = tsne.fit_transform(x)


    # 可视化
    color = ['#F0F8FF', 'green', 'b', 'r', '#7FFFD4', '#FFC0CB', '#00022e']

    # 创建显示的figure
    fig = plt.figure()  # 刷新前一张图片
    ax = plt
    # ax = Axes3D(fig)

    # ax.xlim(-5,5)  # 设置x轴范围
    # ax.ylim(-5,5)  # 设置y轴范围
    ax.scatter(embedded[:, 0], embedded[:, 1], c=[color[t] for t in y], alpha=1, s=np.random.randint(5, 10))


    # 生成三维数据
    # xx = np.random.random(20)*10-5   #取100个随机数，范围在5~5之间
    # yy = np.random.random(20)*10-5
    # X, Y = np.meshgrid(xx, yy)
    # Z = np.sin(np.sqrt(X**2+Y**2))
    # ax.scatter(X, Y, Z, c=np.random.random(400), alpha=0.5, s=np.random.randint(5, 10))

    # 关闭了plot的坐标显示
    # plt.axis('off')

    # plt.show()
    # plt.savefig(os.path.join('tmp', str(epoch) + ".jpg"))
    plt.savefig(os.path.join('openne', 'gcn', 'tmp', str(epoch) + ".jpg"))
