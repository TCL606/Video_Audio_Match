#!/usr/bin/env python

from __future__ import print_function
import random
import time
import os
import numpy as np
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models_exp
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils


#(batchsize,features,sequence)
def get_triplet(vfeat, afeat):
    vfeat_var = vfeat
    afeat_p_var = afeat
    orders = np.arange(vfeat.size(0)).astype('int32')
    negetive_orders = orders.copy()
    for i in range(len(negetive_orders)):
        index_list = list(range(i))
        index_list.extend(list(range(len(negetive_orders))[i + 1:]))
        negetive_orders[i] = index_list[random.randint(
            0,
            len(negetive_orders) - 2)]
    afeat_n_var = afeat[torch.from_numpy(negetive_orders).long()].clone()
    return vfeat_var, afeat_p_var, afeat_n_var


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('int32')
        target1 = torch.from_numpy(label1).long()

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2).long()

        # concat the labels together
        target = torch.cat((target2, target1), 0)

        # transpose the feats
        vfeat0 = vfeat0.transpose(2, 1)
        afeat0 = afeat0.transpose(2, 1)

        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()

        prob = model(vfeat_var, afeat_var)
        loss = criterion(prob, target_var)
        losses.update(loss.item(), vfeat.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                loss=losses)
            print(log_str)
            print(prob[0], prob[opt.batchSize])


def main():
    parser = ArgumentParser()
    parser.add_argument('--config',
                      type=str,
                      help="training configuration",
                      default="./configs/train_config.yaml")
    parser.add_argument('--train_dir',
                      type=str,
                      help="training data directory",
                      default="../Train")
    parser.add_argument('--test_dir',
                      type=str,
                      help="test data directory",
                      default="../Test")

    opts = parser.parse_args()
    assert isinstance(opts, object)
    opt = Config(opts.config)
    print(opt)

    if opt.checkpoint_folder is None:
        opt.checkpoint_folder = 'checkpoints'

    # make dir
    if not os.path.exists(opt.checkpoint_folder):
        os.system('mkdir {0}'.format(opt.checkpoint_folder))

    train_dir = opts.train_dir
    train_dataset = dset(train_dir)

    print('number of train samples is: {0}'.format(len(train_dataset)))
    print('finished loading data')

    opt.manualSeed = random.randint(1, 10000)

    if torch.cuda.is_available() and not opt.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with \"cuda: True\""
        )
        torch.manual_seed(opt.manualSeed)
    else:
        if int(opt.ngpu) == 1:
            print('so we use 1 gpu to training')
            print('setting gpu on gpuid {0}'.format(opt.gpu_id))

            if opt.cuda:
                torch.cuda.set_device(int(opt.gpu_id))
                #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
                torch.cuda.manual_seed(opt.manualSeed)
                cudnn.benchmark = True
    print('Random Seed: {0}'.format(opt.manualSeed))
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=True,
                                               num_workers=int(opt.workers))

    # create model
    model = models_exp.FrameByFrame()

    criterion = torch.nn.CrossEntropyLoss()
    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay**((epoch + 1) // opt.lr_decay_epoch)
    scheduler = LR_Policy(optimizer, lambda_lr)

    for epoch in range(opt.max_epochs):
        train(train_loader, model, criterion, optimizer, epoch, opt)
        scheduler.step()
        if ((epoch + 1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}_state_epoch{1}.pth'.format(
                opt.prefix, epoch + 1)
            utils.save_checkpoint(model.state_dict(), opt.checkpoint_folder,
                                  path_checkpoint)


if __name__ == '__main__':
    main()
