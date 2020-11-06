# train.py
#!/usr/bin/env	python3

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, WarmUpLR

def train(epoch):
    net.train()
    pbar = tqdm(cifar100_training_loader)
    for batch_index, (images, labels) in enumerate(pbar):
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        pbar.set_description("[ EPOCH {epoch} ]".format(epoch=epoch))
        pbar.set_postfix(            
            loss=loss.item(),
            lr= optimizer.param_groups[0]['lr']
        )
    #update training loss for each iteration
    writer.add_scalar('Train/loss', loss.item(), n_iter)
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    test_loss = test_loss / len(cifar100_test_loader.dataset)
    test_acc  = correct.float() / len(cifar100_test_loader.dataset)

    print("[ EPOCH {epoch} ] Loss {loss},  Acc {acc}".format(
        epoch=epoch,
        loss=test_loss / len(cifar100_test_loader.dataset),
        acc=correct.float() / len(cifar100_test_loader.dataset)
        ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return test_loss, test_acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', type=bool, default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    #data preprocessing:
    DATA_ROOT = '/home/bbboming/HDD/Paper/datasets_object/ICIFAR100_70_30/BASE/'
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
    ])
    trainset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), train_transform)
    cifar100_training_loader = torch.utils.data.DataLoader(
        trainset, batch_size = args.b, pin_memory = True,
        num_workers = args.w, shuffle=args.s
    )

    test_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
    ])

    testset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'test'), test_transform)
    cifar100_test_loader = torch.utils.data.DataLoader(
        testset, batch_size = args.b, pin_memory = True,
        num_workers = args.w, shuffle=False
    )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', threshold=1e-7)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):

        train(epoch)
        test_loss, test_acc = eval_training(epoch)

        #Write Lr Log
        writer.add_scalars("Lr", {'lr': get_lr(optimizer)}, epoch+1)

        #Train Scheduler
        train_scheduler.step(test_loss)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < test_acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = test_acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
