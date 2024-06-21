
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.vgg import *
from conf import settings
from utils import get_network,  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM

# 指定保存数据的文件夹路径
folder_path = r'C:\Users\Raytrack\Documents\MR_fussion\MF\huatu_data'

my_train_loss = []
my_eval_loss = []
my_train_correct= []
my_eval_correct = []
def train(epoch):
    start = time.time()
    train_loss = 0.0 # cost function error
    correct = 0.0
    net.train()
    net3.train()
    p=0
    for i, x in enumerate(CESM_10_train_l):
        low_energy = x['LOW_ENERGY']
        high_energy = x['HIGH_ENERGY']
        enhance = x['ENHANCE']
        
        labels=x['label']
        labels = torch.LongTensor(labels.numpy())
        # labels = torch.IntTensor(labels).to(torch.long)

        if args.gpu:

            low_energy = low_energy.cuda()
            high_energy = high_energy.cuda()
            enhance = enhance.cuda()
            labels=labels.cuda()

        optimizer.zero_grad()
        outputs_low_energy,ou_CCL = net(low_energy)
        outputs_high_energy,ou_CCR = net(high_energy)
        outputs_enhance,ou_MLOL = net(enhance)

        outputs1 = torch.cat((outputs_low_energy, outputs_high_energy,outputs_enhance), dim=1)

        ou11=net3(outputs1)
        loss = loss_function(ou11, labels)
        print('loss:{}'.format(
            loss.item()
        )

        )



        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = ou11.max(1)
        
        correct += preds.eq(labels).sum()
        n_iter = (epoch - 1) * len(CESM_10_train_l) + i + 1

        train_correct=correct.float() / (len(CESMdata))
        my_train_loss.append(loss.item())
        my_train_correct.append(train_correct)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            # correct.float() / len(CESMdata),
            epoch=epoch,
            trained_samples=i * args.b + len(low_energy),
            total_samples=len(CESMdata)
        ))


        if epoch <= args.warm:
            warmup_scheduler.step()


    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Train Average loss: {:.4f}\tTrain Accuarcy: {:0.6f}'.format(
        train_loss / len(CESMdata),
        correct.float() / (len(CESMdata))
        ))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    net3.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0

    for i, x in enumerate(CESM_10_test_l):

        low_energy = x['LOW_ENERGY']
        high_energy = x['HIGH_ENERGY']
        enhance = x['ENHANCE']
        
        labels=x['label']
        labels = torch.LongTensor(labels.numpy())
        # labels = torch.IntTensor(labels).to(torch.long)

        if args.gpu:

            low_energy = low_energy.cuda()
            high_energy = high_energy.cuda()
            enhance = enhance.cuda()
            labels=labels.cuda()

        optimizer.zero_grad()
        outputs_low_energy,ou_CCL = net(low_energy)
        outputs_high_energy,ou_CCR = net(high_energy)
        outputs_enhance,ou_MLOL = net(enhance)


        outputs1 = torch.cat((outputs_low_energy, outputs_high_energy,outputs_enhance), dim=1)

        ou11=net3(outputs1)
        loss = loss_function(ou11, labels)


        n_iter = (epoch - 1) * len(CESM_10_test_l) + i + 1
        test_loss += loss.item()
        _, preds = ou11.max(1)
        correct += preds.eq(labels).sum()

        eval_correct= correct.float() / (len(CESMdata2))
        my_eval_loss.append(loss.item())
        my_eval_correct.append(loss.item())

    finish = time.time()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Eval set: Epoch: {},Eval Average loss: {:.4f},Eval Accuracy: {:.4f} Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(CESMdata2),
        correct.float() / (len(CESMdata2)),
        finish - start
    ))
    print()


    return correct.float() / (len(CESMdata2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = vgg16_bn()

    net = net.cuda()

    # path1 = 'pretrained models'
    # net.load_state_dict(torch.load(path1),strict=False)
    net3=nn.Linear(512*3,2)
    net3.cuda()





    CESMdata = CESM(base_dir='./MF/h5py/train',transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       transforms.RandomRotation(10), # 随机旋转-10°~10°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),

                   ]))
    CESM_10_train_l = DataLoader(CESMdata, batch_size=args.b, shuffle=True, drop_last=False,
                                 pin_memory=torch.cuda.is_available())

    CESMdata2 = CESM(base_dir='./MF/h5py/valid',transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    CESM_10_test_l = DataLoader(CESMdata2, batch_size=args.b, shuffle=False, drop_last=False,
                                 pin_memory=torch.cuda.is_available())






    loss_function = nn.CrossEntropyLoss()
    loss_function.cuda()

    optimizer = optim.SGD([{"params":net.parameters()},{"params":net3.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(CESM_10_train_l)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'vgg16', recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'vgg16', settings.TIME_NOW)



    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_acc2=0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue


        train(epoch)
        acc= eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch <= settings.MILESTONES[3] and best_acc < acc:
            weights_path = checkpoint_path.format(net='vgg16', epoch=epoch, type='best')
            weights_path3 = checkpoint_path.format(net='net3', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            torch.save(net3.state_dict(), weights_path3)
            best_acc = acc

            continue

    # 指定保存数据的文件夹路径
    folder_path = folder_path

    # 保存 label 和 pro 到一个文件
    np.savez(os.path.join(folder_path, 'loss.npz'), my_train_loss=my_train_loss, my_eval_loss=my_eval_loss,
             my_train_correct=my_train_correct,my_eval_correct =my_eval_correct )
    print('my_train_loss:',my_train_loss)
    print('my_eval_loss:',my_eval_loss)




