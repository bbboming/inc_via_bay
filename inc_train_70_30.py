#test.py
#!/usr/bin/env python3

import os
import datetime
import math
import argparse
import json
import numpy as np
from timeit import default_timer
from matplotlib import pyplot as plt

import GPyOpt
import GPy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import WarmUpLR
from models.resnet import resnet50

csv_folder = "/home/bbboming/GDrive/Yonsei/논문/APSIPA/start_2/70_30excel"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
csv_file = os.path.join(csv_folder,"%s.csv"%(datetime.datetime.now().isoformat()))
csv = open(csv_file,"w")

checkpoint = '/home/bbboming/HDD/Paper/inc_with_bay/pytorch-cifar100/checkpoint/resnet50/2020-05-16T18:25:40.913847/resnet50-192-best.pth'
class layer_sharing_acc():
    def __init__(self, acc_0=1, acc_53=0):
        self.basenet = resnet50().cuda()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.basenet = torch.nn.DataParallel(self.basenet)
            cudnn.benchmark = True
        
        self.basenet.load_state_dict(torch.load(checkpoint), True)
        self.basenet.eval()
        
        self.count = 0
        self.total_wc = 0
        for m in self.basenet.modules():
            if isinstance(m, nn.Conv2d):
                self.count += 1
                for param in m.parameters():
                    self.total_wc += param.numel()

        self.batch_size = 128
        self.shuffle = True
        self.lr = 0.1
        self.warm = 1
        self.min_acc = acc_53
        self.max_acc = acc_0
        self.iteration = 0


    def f(self,x, return_acc=False): #x, layer number to calculate

        if x.size == 1:
            x = np.append(x, 0.32)
        x = x.reshape(1,2)

        target = int(x[:,0])
        start_time = default_timer()

        self.net = resnet50().cuda()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        
        self.net.load_state_dict(torch.load(checkpoint), True)
        self.net.module.fc = nn.Linear(512 * 4, 30).cuda()
        self.net.train()


        cur_wc = 0
        count = 0
        for m in self.net.modules():
            if target == count:
                break
            elif isinstance(m, nn.Conv2d):
                for param in m.parameters():
                    cur_wc += param.numel()
                    param.requires_grad = False
            elif isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = False
                count += 1

        BASE_DATA_ROOT = '/home/bbboming/HDD/Paper/datasets_object/ICIFAR100_70_30/BASE/'
        DATA_ROOT = '/home/bbboming/HDD/Paper/datasets_object/ICIFAR100_70_30/INC1/'
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
        ])
        trainset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), train_transform)
        cifar100_training_loader = torch.utils.data.DataLoader(
            trainset, batch_size = self.batch_size, pin_memory = True,
            num_workers = 4, shuffle=self.shuffle
        )
        test_transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
        ])
        testset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'test'), test_transform)
        cifar100_test_loader = torch.utils.data.DataLoader(
            testset, batch_size = self.batch_size, pin_memory = True,
            num_workers = 4, shuffle=False
        )
        base_testset = datasets.ImageFolder(os.path.join(BASE_DATA_ROOT, 'test'), test_transform)
        cifar100_base_test_loader = torch.utils.data.DataLoader(
            base_testset, batch_size = self.batch_size, pin_memory = True,
            num_workers = 4, shuffle=False
        )

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        iter_per_epoch = len(cifar100_training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.warm)
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'resnet50_inc', settings.TIME_NOW)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{target}-{type}.pth')

        best_acc = 0.0
        best_base_acc = 0.0
        best_inc_acc  = 0.0
        for epoch in range(1, settings.EPOCH):
            self.net.train()
            for batch_index, (images, labels) in enumerate(cifar100_training_loader):
                images = Variable(images)
                labels = Variable(labels)

                labels = labels.cuda()
                images = images.cuda()

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                if epoch <= self.warm:
                    warmup_scheduler.step()

                n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
                
            #print('[Target {target}] [Training Epoch: {epoch}/{total_epoch}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            #    loss.item(),
            #    optimizer.param_groups[0]['lr'],
            #    target=target,
            #    epoch=epoch,
            #    total_epoch=settings.EPOCH
            #))


            #Evaluation Accuracy
            self.net.eval()
            self.basenet.eval()

            test_loss = 0.0 # cost function error
            correct   = 0.0


            #INC Testset 
            for (images, labels) in cifar100_test_loader:
                images = Variable(images)
                labels = Variable(labels)
                images = images.cuda()
                labels = labels.cuda()
                
                soft_layer = nn.Softmax(dim=1).cuda()

                base_outputs = self.basenet(images)
                outputs      = self.net(images)

                loss = loss_function(outputs, labels)
                test_loss += loss.item()

                soft_base = soft_layer(base_outputs)
                soft_inc  = soft_layer(outputs)
                softmax = torch.cat([soft_base, soft_inc], dim=1)
                labels_all = labels + 70
                _, preds = softmax.max(1)
                correct += preds.eq(labels_all).sum()

            #Base Testset
            correct_base = 0.0
            for (images, labels) in cifar100_base_test_loader:
                images = Variable(images)
                labels = Variable(labels)
                images = images.cuda()
                labels = labels.cuda()
                
                soft_layer = nn.Softmax(dim=1).cuda()

                base_outputs = self.basenet(images)
                outputs      = self.net(images)

                soft_base = soft_layer(base_outputs)
                soft_inc  = soft_layer(outputs)
                softmax = torch.cat([soft_base, soft_inc], dim=1)
                labels_all = labels
                _, preds = softmax.max(1)
                correct_base += preds.eq(labels_all).sum()            
            

            avg_loss = test_loss / len(cifar100_test_loader.dataset)
            base_acc = correct_base.float() / len(cifar100_base_test_loader.dataset)
            inc_acc  = correct.float() / len(cifar100_test_loader.dataset)
            acc = (correct.float()+ correct_base.float()) / (len(cifar100_test_loader.dataset) + len(cifar100_base_test_loader.dataset))

            train_scheduler.step(avg_loss)

            #start to save best performance model after learning rate decay to 0.01 
            if epoch > 10 and best_acc < acc:
                torch.save(self.net.state_dict(), checkpoint_path.format(target=target, net='resnet50', type='best'))
                best_acc = acc
                best_base_acc = base_acc
            
            if epoch > 10 and best_inc_acc < inc_acc:
                best_inc_acc = inc_acc

        memory_efficiency = cur_wc / self.total_wc
        obj_acc = best_acc.detach().cpu().item()
        alpha = x[:,1].item()
        threshold = 0.02
        # mem_target = 0.8
        obj_f = np.abs((self.max_acc- obj_acc)-threshold)
        print("obj_acc %f, max_acc %f obj_f %f"%(obj_acc, self.max_acc, obj_f))
        print_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " x= {x}, alpha= {alpha} Memory_Efficiency= {memory_efficiency}, combined_classification_acc= {best_acc}, obj_acc= {obj_acc}, OBJ_F= {obj_f}" \
                        .format(x=target, alpha=alpha,best_acc=best_acc, obj_acc=obj_acc, memory_efficiency=memory_efficiency, obj_f=obj_f)
        with open("history.log", "a") as f_hist:
            f_hist.write(print_str + "\n")
        print(print_str)
       
        if self.min_acc != 0:
           csv.write("%d, %d, %f, %f, %f, %f, %f\n"%(self.iteration, target, obj_acc, threshold, obj_f, self.min_acc, self.max_acc))
           self.iteration += 1


        end_time = default_timer()
        # print("operation time: ",(end_time - start_time))

        os.system("cp -f acc.json acc_old.json")
        
        if return_acc: 
            return (best_acc.detach().cpu().item())
        return (obj_f)

def estimate_sharing_layer_count(acc_0=1, acc_53=0):
    func = layer_sharing_acc(acc_0, acc_53)

    space = [{'name': 'var_1', 'type': 'discrete','domain': range(1,54)},
             {'name': 'var_2', 'type': 'continuous','domain': (0,1)}]
    space2 = [{'name': 'var_1', 'type': 'discrete','domain': range(1,54)}]
    # feasible_region = GPyOpt.Design_space(space=space)
    feasible_region = GPyOpt.Design_space(space=space2)

    initial_design = GPyOpt.experiment_design.initial_design('random',feasible_region,5)
 
    objective = GPyOpt.core.task.SingleObjective(func.f)

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=acquisition_optimizer)

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    max_time = None
    tolerance = 1
    max_iter = 30

    with open("history.log", "a") as f_hist:
        f_hist.write("---------------------------------------------------------------\n")
        f_hist.write("INC(70/30) Result\n")
        f_hist.write("---------------------------------------------------------------\n")
    
    csv.write("iteration, layer, comb_acc, threshold, obj_f, min_acc, max_acc\n")

    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

    bo.plot_acquisition(filename="%s_1.png"%csv_file)
    bo.plot_convergence(filename="%s_2.png"%csv_file)

    plot_acquisition_max(bo)
    plot_convergence_max(bo)

    sharing_layer_count = int(bo.x_opt)

    return sharing_layer_count

if __name__ == "__main__":
    obj = layer_sharing_acc()
    acc_0 = obj.f(np.array([0]), return_acc=True)
    acc_53 = obj.f(np.array([53]), return_acc=True)
    print(estimate_sharing_layer_count(acc_0, acc_53))
