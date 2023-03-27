import unittest


# 处理训练集数据
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import data


# 处理训练集数据
def train_data_process():
    # 加载FashionMNIST数据集
    train_data = FashionMNIST(root="data",  # 数据路径
                              train=True,  # 只使用训练数据集
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              # AlexNet的输入数据大小为224*224，因此这里将FashionMNIST数据集的尺寸从28扩展到224
                              # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型
                              download=True,  # 若本身没有下载相应的数据集，则选择True
                              )
    train_loader = Data.DataLoader(dataset=train_data,  # 传入的数据集
                                   batch_size=64,  # 每个Batch中含有的样本数量
                                   shuffle=True,  # 不对数据集重新排序
                                   num_workers=0,  # 加载数据所开启的进程数量
                                   )
    print("The number of batch in train_loader:", len(train_loader))  # 一共有938个batch，每个batch含有64个训练样本

    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    class_label = train_data.classes  # 训练集的标签
    class_label[0] = "T-shirt"
    print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224
    #初始化一个列表
    a = []
    # 可视化一个Batch的图像
    plt.figure(figsize=(12, 5))
    for ii in np.arange(2):
        plt.subplot(1, 2, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()
    x = batch_x[0]

    print(x.shape)
    return train_loader, class_label


class MyTestCase(unittest.TestCase):
    def test_something(self):
        train_data_process()


if __name__ == '__main__':
    unittest.main()
