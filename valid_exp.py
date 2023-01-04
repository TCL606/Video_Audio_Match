import models_exp
import torch
import numpy as np
import os
from models.vakm_model import VAKMModel

epoch = 10
# model = models_exp.FrameByFrame()
# ckpt = torch.load('checkpoints/VA_METRIC_state_epoch{}.pth'.format(epoch), map_location='cpu')
# model.load_state_dict(ckpt)
model = torch.load('/root/bqqi/changli/STD2022/output/debug/Debug_state_epoch_10.pth')
model.cuda().eval()

vpath = '../Train/vfeat'
apath = '../Train/afeat'
valid_npy = np.load('/root/bqqi/changli/STD2022/data/valid_334.npy')

def gen_tsample(n):
    tsample = np.zeros((500, n)).astype(np.int16)
    for i in range(500):
        tsample[i] = np.random.permutation(num)[:n]
    np.save('tsample_{}.npy'.format(n), tsample)

def get_top(tsample, rst):
    top1 = 0.0
    top5 = 0.0
    n = tsample.shape[1]
    for i in range(500):
        idx = tsample[i]
        rsti = rst[idx][:,idx]
        assert rsti.shape[0] == n
        assert rsti.shape[1] == n
        sorti = np.sort(rsti, axis=1)
        for j in range(n):
            if rsti[j,j] == sorti[j,-1]:
                top1 += 1
            if rsti[j,j] >= sorti[j,-5]:
                top5 += 1
    top1 = top1 / 500 / n
    top5 = top5 / 500 / n
    print('Top1 accuracy for sample {} is: {}.'.format(n, top1))
    print('Top5 accuracy for sample {} is: {}.'.format(n, top5))

def valid(valid_dir, model, valid_idx, epoch):
    model.cuda().eval()
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
        if True:
            model.to('cuda')
            aemb, vemb = model(afeats.cuda(), vfeats.cuda(), None)

    from tqdm import tqdm 
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
    np.save('valid_rst.npy', rst)
    print("=========================================")
    print(f"top1 acc: {top1_acc}")
    print(f"top5 acc: {top5_acc}")
    print(f"top50 acc: {top50_acc}")
    print("=========================================")
    return rst

# num = len(valid)
# rst = np.zeros((num, num))
# vfeats = torch.zeros(num, 512, 10).float()
# afeats = torch.zeros(num, 128, 10).float()
# for i in range(num):
#     vfeat = np.load(os.path.join(vpath, '%04d.npy'%valid[i]))
#     for j in range(num):
#         vfeats[j] = torch.from_numpy(vfeat).float().permute(1, 0)
#         afeat = np.load(os.path.join(apath, '%04d.npy'%valid[j]))
#         afeats[j] = torch.from_numpy(afeat).float().permute(1, 0)
#     with torch.no_grad():
#         out = model(vfeats.cuda(), afeats.cuda())
#     rst[i] = (out[:,1] - out[:,0]).cpu().numpy()
#     print(i)

# np.save('rst_epoch{}.npy'.format(epoch), rst)


# print('Test checkpoint epoch {}.'.format(epoch))
# rst = np.load('rst_epoch30.npy')

num = len(valid_npy)
rst = valid('../Train', model, valid_npy, epoch)
gen_tsample(50)

tsample = np.load('tsample_{}.npy'.format(50))
get_top(tsample, rst)






