from fairseq.modules import TransformerEncoderLayer
from yaml import FullLoader
import yaml
from argparse import ArgumentParser, Namespace
import yaml
from yaml import SafeLoader, SafeDumper
from argparse import Namespace
import torch
import torch.nn as nn
from models.va_model import *
from models.transfomer import *
import numpy as np
from data.va_dataset import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy
from tools import utils
import time
from tqdm import tqdm, trange

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, cfg, pbar=None):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()
    for step_in_epoch, batch in enumerate(train_loader):
        index, afeat, vfeat = batch['index'], batch['afeat'], batch['vfeat']
        aemb, vemb = model(afeat, vfeat)
        
        # negative sampling
        with torch.no_grad():
            neg_samples = model.get_neg_samples(index)
            neg_sample_afeats = model.load_neg_samples(neg_samples)
            neg_aembs = model.extract_audio_feature(neg_sample_afeats)
            total_aembs = torch.stack([torch.cat([aemb[i: i + 1], neg_aembs], dim=0) for i in range(aemb.shape[0])])

        # compute loss
        probs = model.get_probs(total_aembs, vemb)
        labels = torch.zeros(probs.shape[0], dtype=torch.long).to(probs.device)
        loss = criterion(probs, labels)

        losses.update(loss.item(), vfeat.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step_in_epoch += 1
        if pbar is not None:
            pbar.update(1)
    log_str = f' Epoch: [{epoch}][{step_in_epoch}/{len(train_loader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {losses.val:.4f} ({losses.avg:.4f})'
    print(log_str)

def train(args):
    model_cfg = yaml.load(open(args.model_config), Loader=SafeLoader)
    train_cfg = yaml.load(open(args.train_config), Loader=SafeLoader)
    assert model_cfg['dataset']['train_dir'] == train_cfg['train_dir']
    assert model_cfg['dataset']['test_dir'] == train_cfg['test_dir']

    # make dir
    if not os.path.exists(train_cfg['output_dir']):
        import platform
        sys_platform = platform.platform().lower()
        if "windows" in sys_platform:
            os.system(f'md {train_cfg["output_dir"]}')
        elif 'linux' in sys_platform:
            os.system(f'mkdir {train_cfg["output_dir"]}')

    train_dir = model_cfg['dataset']['train_dir']
    train_dataset = VADataset(train_dir)

    print('number of train samples is: {0}'.format(len(train_dataset)))

    torch.manual_seed(train_cfg['seed'])
    if torch.cuda.is_available() and not train_cfg['cuda']:
        print(
            "WARNING: You have a CUDA device, so you should probably run with \"cuda: True\""
        )
    else:
        if train_cfg['cuda']:
            torch.cuda.set_device(int(train_cfg['gpu_id']))
            #os.environ['CUDA_VISIBLE_DEVICES'] = train_cfg.gpu_id
            cudnn.benchmark = True

    print(f'Random Seed: {train_cfg["seed"]}')

    # train data loader
    train_loader = VADataloader(train_dataset,
                                batch_size=train_cfg['batchSize'],
                                shuffle=True,
                                num_workers=int(train_cfg['workers']))

    # create model
    model = VAModel(model_cfg)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    if train_cfg['cuda']:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), train_cfg['lr'])

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: train_cfg['lr_decay']**((epoch + 1) // train_cfg['lr_decay_epoch'])
    scheduler = LR_Policy(optimizer, lambda_lr)

    total_steps = train_cfg['max_epochs'] * len(train_loader)
    pbar = tqdm(total=total_steps, leave=False)
    while pbar.n < total_steps:
        for epoch in range(train_cfg['max_epochs']):
            train_one_epoch(train_loader, model, criterion, optimizer, epoch, train_cfg, pbar)
            scheduler.step()
            if ((epoch + 1) % train_cfg['epoch_save']) == 0:
                path_checkpoint = os.path.join(train_cfg['output_dir'], f'{train_cfg["prefix"]}_state_epoch_{epoch + 1}.pth')
                # utils.save_checkpoint(model.state_dict(), train_cfg['output_dir'], path_checkpoint)
                torch.save(model, path_checkpoint)
                print("save ckpt at " + path_checkpoint)
    path_checkpoint = os.path.join(train_cfg['output_dir'], f'{train_cfg["prefix"]}_state_epoch_last.pth')
   # utils.save_checkpoint(model.state_dict(), train_cfg['output_dir'], path_checkpoint)
    torch.save(model, path_checkpoint)
    print("save ckpt at " + path_checkpoint)

def test(args):
    model = torch.load(args.ckpt_path)
    dataset = VADataset(args.test_dir)
    dataloader = VADataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
    available_sample_num = len(dataset)
    samples = np.array(range(available_sample_num))

    top1_correct = 0
    top5_correct = 0
    model.eval()
    with torch.no_grad():    
        sample_afeats = model.load_neg_samples(samples)
        total_aembs = model.extract_audio_feature(sample_afeats).unsqueeze(0)

        for batch in tqdm(dataloader):
            index, afeat, vfeat = batch['index'], batch['afeat'], batch['vfeat']
            vemb = model.extract_video_feature(vfeat)
            probs = model.get_probs(total_aembs, vemb)
            top1 = torch.argmax(probs, dim=1)
            top5 = torch.topk(probs, 5, dim=1).indices
            for i in range(len(index)):
                top1_correct += index[i] == top1[i]
                top5_correct += index[i] in top5[i]

    print(f'top1 acc: {top1_correct / available_sample_num}')
    print(f'top5 acc: {top5_correct / available_sample_num}')

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_config',
                      type=str,
                      help="training configuration",
                      default="./configs/va_model.yaml")

    parser.add_argument('--train_config',
                        type=str,
                        help="training configuration",
                        default="./configs/trainva_config.yaml")
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        help='path to checkpoint', 
                        default='./output/debug/VA_MODEL_state_epoch_last.pth')
    parser.add_argument('--test_dir',
                        type=str,
                        help='test data directory',
                        default='../Test/Clean')

    args = parser.parse_args()
    if args.train:
        train(args)

    if args.test:
        test(args)

if __name__ == '__main__':
    main()
