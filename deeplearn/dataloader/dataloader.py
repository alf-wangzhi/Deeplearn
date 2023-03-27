import os
import math
import argparse


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import transforms

from dataloader.utils import *
from dataloader.my_dataset import *



def dataloador(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")



    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                   #transforms.CenterCrop(96),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)



    for i,(X,X_label) in enumerate(train_loader):
        if i==0:

            break

    return train_loader,val_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"C:\Users\wangzhiqiang\Desktop\deep-learning-for-image-processing-master\pytorch_classification\vision_transformer\flower_photos")
    parser.add_argument('--batch-size', type=int, default=8)
    opt = parser.parse_args()

    dataloador(opt)
