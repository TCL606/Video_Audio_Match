from yaml import FullLoader
import yaml
from argparse import ArgumentParser, Namespace
from yaml import SafeLoader, SafeDumper
import torch
import torch.nn as nn
from models.va_model import *
from models.vakm_model import *
import numpy as np
from data.va_dataset import *
from data.vakm_dataset import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy
from tools import utils
import time
import wandb
from tqdm import tqdm, trange

def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch, cfg, pbar=None):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    model.train()
    if epoch <= cfg['kmeans_epoch']:
        alpha = 1
        beta = 1
    else:
        alpha = 1
        beta = 0
    end = time.time()
    for step_in_epoch, batch in enumerate(train_loader):
        index, afeat, vfeat, pos_ki = batch['index'], batch['afeat'], batch['vfeat'], batch['kmeans_id']
        neg_sample_num = cfg['neg_sample_num']
        neg_afeat, neg_ki = train_loader.load_neg_afeat(index, neg_sample_num)
        if cfg['cuda']:
            afeat = afeat.cuda()
            vfeat = vfeat.cuda()
            neg_afeat = neg_afeat.cuda()
        logits, labels, k_preds, k_labels = model(afeat, vfeat, neg_afeat, pos_ki, neg_ki)
        loss1 = criterion(logits, labels)
        loss2 = nn.functional.cross_entropy(k_preds, k_labels)
        loss = alpha * loss1 + beta * loss2
        
        losses.update(loss.item(), vfeat.size(0))
        losses1.update(loss1.item(), vfeat.size(0))
        losses2.update(loss2.item(), vfeat.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step_in_epoch += 1
        if pbar is not None:
            pbar.update(1)

        if (pbar.n + 1) % cfg['step_log'] == 0:
            log_str = f' Epoch: [{epoch}][{step_in_epoch}/{len(train_loader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {losses.val:.4f} ({losses.avg:.4f})'
            if args.wandb:
                wandb.log({
                    'loss': losses.avg,
                    'loss1': losses1.avg,
                    'loss2': losses2.avg
                })
            print(log_str)
            losses.reset()

    log_str = f' Epoch: [{epoch}][{step_in_epoch}/{len(train_loader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {losses.val:.4f} ({losses.avg:.4f})'
    print(log_str)

def train(args):
    model_cfg = yaml.load(open(args.model_config), Loader=SafeLoader)
    train_cfg = yaml.load(open(args.train_config), Loader=SafeLoader)

    # make dir
    if not os.path.exists(train_cfg['output_dir']):
        import platform
        sys_platform = platform.platform().lower()
        if "windows" in sys_platform:
            os.system(f'md {train_cfg["output_dir"]}')
        elif 'linux' in sys_platform:
            os.system(f'mkdir {train_cfg["output_dir"]}')

    train_dir = train_cfg['train_dir']
    valid_dir = train_cfg['valid_dir']
    train_npy = train_cfg['train_npy']
    valid_npy = train_cfg['valid_npy']
    # train_dataset = VADataset(train_dir, npy=train_npy, split='train')
    train_dataset = VAKMDataset(train_dir, npy=train_npy, kmeans_npy=train_cfg['kmeans_npy'])

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
    train_loader = VAKMDataloader(train_dataset,
                                batch_size=train_cfg['batchSize'],
                                shuffle=True,
                                num_workers=int(train_cfg['workers']))

    # create model
    model = VAKMModel(model_cfg)

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
    
    
    name = 'debug' if 'wandb' not in train_cfg else train_cfg['wandb'] 
    if args.wandb:
        wandb.init(project='std2022', name=name, config=train_cfg, entity='tcl606')

    while pbar.n < total_steps:
        for epoch in range(train_cfg['max_epochs']):
            train_one_epoch(args, train_loader, model, criterion, optimizer, epoch, train_cfg, pbar)
            scheduler.step()
            if ((epoch + 1) % train_cfg['epoch_save']) == 0:
                path_checkpoint = os.path.join(train_cfg['output_dir'], f'{train_cfg["prefix"]}_state_epoch_{epoch + 1}.pth')
                torch.save(model, path_checkpoint)
                print("save ckpt at " + path_checkpoint)
            if ((epoch + 1) % train_cfg['epoch_valid']) == 0:
                valid(args, valid_dir=valid_dir, npy=valid_npy, model=model, gpu=train_cfg['cuda'])
    path_checkpoint = os.path.join(train_cfg['output_dir'], f'{train_cfg["prefix"]}_state_epoch_last.pth')
    torch.save(model, path_checkpoint)
    print("save ckpt at " + path_checkpoint)

def valid(args, valid_dir, npy, model=None, ckpt_path=None, gpu=False):
    if model == None:
        model = torch.load(ckpt_path)
    if npy is not None:
        valid_idx = np.load(npy)
    else:
        valid_idx = np.arange(339)
    model.eval()
    vpath = os.path.join(valid_dir, 'vfeat')
    apath = os.path.join(valid_dir, 'afeat')
    num = len(valid_idx)
    rst = np.zeros((num, num))
    top1_acc = 0
    top5_acc = 0
    top50_acc = 0

    vfeats = torch.zeros(num, 10, 512).float()
    afeats = torch.zeros(num, 10, 128).float()
    for j in range(num):
        vfeat = np.load(os.path.join(vpath, '%04d.npy' % (valid_idx[j])))
        vfeats[j] = torch.from_numpy(vfeat).float()
        afeat = np.load(os.path.join(apath, '%04d.npy' % (valid_idx[j])))
        afeats[j] = torch.from_numpy(afeat).float()
    with torch.no_grad():
        if gpu:
            model.to('cuda')
            aemb, vemb = model(afeats.cuda(), vfeats.cuda(), None)
        else:
            model.to('cpu')
            aemb, vemb = model(afeats, vfeats, None)

    for i in tqdm(range(num)):
        with torch.no_grad():
            out = torch.cosine_similarity(vemb[i], aemb, dim=1)
        top1_acc += i in torch.topk(out, 1).indices
        top5_acc += i in torch.topk(out, 5).indices
        top50_acc += i in torch.topk(out, 50).indices
        rst[i] = out.cpu().numpy()
    top1_acc /= num
    top5_acc /= num
    top50_acc /= num
    # np.save('valid_rst.npy', rst)
    print("=========================================")
    print(f"top1 acc: {top1_acc}")
    print(f"top5 acc: {top5_acc}")
    print(f"top50 acc: {top50_acc}")
    print("=========================================")
    if args.wandb:
        wandb.log({
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'top50_acc': top50_acc
        })

def test(args):
    model = torch.load(args.ckpt_path)
    model.eval()
    vpath = os.path.join(args.test_dir, 'vfeat')
    apath = os.path.join(args.test_dir, 'afeat')
    rst = np.zeros((804, 804))
    top1 = np.zeros((804, 1))
    top5 = np.zeros((804, 5))
    top50 = np.zeros((804, 50))

    vfeats = torch.zeros(804, 10, 512).float()
    afeats = torch.zeros(804, 10, 128).float()
    for j in range(804):
        vfeat = np.load(os.path.join(vpath, '%04d.npy' % j))
        vfeats[j] = torch.from_numpy(vfeat).float()
        afeat = np.load(os.path.join(apath, '%04d.npy' % j))
        afeats[j] = torch.from_numpy(afeat).float()
    with torch.no_grad():
        if args.gpu:
            model.to('cuda')
            aemb, vemb = model(afeats.cuda(), vfeats.cuda(), None)
        else:
            model.to('cpu')
            aemb, vemb = model(afeats, vfeats, None)

    for i in tqdm(range(804)):
        with torch.no_grad():
            out = torch.cosine_similarity(vemb[i], aemb, dim=1).cpu()
        top1[i] = torch.topk(out, 1).indices
        top5[i] = torch.topk(out, 5).indices
        top50[i] = torch.topk(out, 50).indices
        rst[i] = out.numpy()
    np.save('test_rst.npy', rst)
    np.save('test_top1.npy', top1)
    # np.save('test_top5.npy', top5)
    # np.save('test_top50.npy', top50)

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
    parser.add_argument('--valid', action='store_true', help='valid the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        help='path to checkpoint', 
                        default='./output/debug/KMeans5e-4_state_epoch_last.pth')
    parser.add_argument('--valid_dir',
                        type=str,
                        help='valid data directory',
                        default='../Train')
    parser.add_argument('--valid_npy',
                        type=str,
                        help='valid numpy idx',
                        default='./data/valid_334.npy')
    parser.add_argument('--test_dir',
                        type=str,
                        help='test data directory',
                        default='../Test/Noise')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    args = parser.parse_args()
    if args.train:
        train(args)

    if args.valid:
        valid(args, valid_dir=args.valid_dir, npy=args.valid_npy, model=None, ckpt_path=args.ckpt_path, gpu=args.gpu)

    if args.test:
        test(args)

if __name__ == '__main__':
    main()
