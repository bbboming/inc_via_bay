#test.py
#!/usr/bin/env python3

import os
import argparse
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from models.resnet import resnet50

from conf import settings

if __name__ == '__main__':
    checkpoint = '/home/bbboming/HDD/Paper/inc_with_bay/pytorch-cifar100/checkpoint/resnet50/2020-05-16T18:25:40.913847/resnet50-192-best.pth'
    BASE_DATA_ROOT = '/home/bbboming/HDD/Paper/datasets_object/ICIFAR100_70_30/BASE/'
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    basenet = resnet50().cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        basenet = torch.nn.DataParallel(basenet)
        cudnn.benchmark = True
        
    basenet.load_state_dict(torch.load(checkpoint), True)
    # basenet.module.fc = nn.Linear(512 * 4, 30).cuda()
    basenet.eval()

    test_transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
        ])

    
    base_testset = datasets.ImageFolder(os.path.join(BASE_DATA_ROOT, 'test'), test_transform)
    cifar100_test_loader = torch.utils.data.DataLoader(
        base_testset, batch_size = args.b, pin_memory = True,
        num_workers = 4, shuffle=False
    )
    
    print(basenet)
    basenet.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = basenet(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1 
        correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in basenet.parameters())))
