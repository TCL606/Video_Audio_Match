import models_exp
import torch
import numpy as np
import os
from argparse import ArgumentParser
from transformers import TrainingArguments
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--test_dir',
                        type=str,
                        help="test data directory",
                        default="../Test")
    parser.add_argument('--test_ckpt',
                        type=str,
                        help="test checkpoint file",
                        default="./output/train/VA_METRIC/VA_METRIC_state_epoch100.pth")
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    parser.add_argument('--output_dir',
                        type=str,
                        help='output directory',
                        default='./output/test')

    args = parser.parse_args()

    epoch = 100
    model = models_exp.FrameByFrame()
    ckpt = torch.load(args.test_ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    if args.gpu:
        model.cuda()

    vpath = os.path.join(args.test_dir, 'vfeat')
    apath = os.path.join(args.test_dir, 'afeat')

    def gen_tsample(n):
        tsample = np.zeros((500, n)).astype(np.int16)
        for i in range(500):
            tsample[i] = np.random.permutation(804)[:n]
        np.save('tsample_{}.npy'.format(n), tsample)

    def get_top(tsample, rst):
        top1 = 0.0
        top5 = 0.0
        n = tsample.shape[1]
        for i in range(500):
            idx = tsample[i]
            rsti = rst[idx][:, idx]
            assert rsti.shape[0] == n
            assert rsti.shape[1] == n
            sorti = np.sort(rsti, axis=1)
            for j in range(n):
                if rsti[j, j] == sorti[j, -1]:
                    top1 += 1
                if rsti[j, j] >= sorti[j, -5]:
                    top5 += 1
        top1 = top1 / 500 / n
        top5 = top5 / 500 / n
        print('Top1 accuracy for sample {} is: {}.'.format(n, top1))
        print('Top5 accuracy for sample {} is: {}.'.format(n, top5))

    rst = np.zeros((804, 804))
    vfeats = torch.zeros(804, 512, 10).float()
    afeats = torch.zeros(804, 128, 10).float()
    top1_acc = 0
    top5_acc = 0
    top50_acc = 0
    for i in tqdm(range(804)):
        vfeat = np.load(os.path.join(vpath, '%04d.npy' % i))
        for j in range(804):
            vfeats[j] = torch.from_numpy(vfeat).float().permute(1, 0)
            afeat = np.load(os.path.join(apath, '%04d.npy' % j))
            afeats[j] = torch.from_numpy(afeat).float().permute(1, 0)
        with torch.no_grad():
            if args.gpu:
                vfeats = vfeats.cuda()
                afeats = afeats.cuda()
            out = model(vfeats, afeats)
        temp = (out[:, 1] - out[:, 0]).cpu()
        top1_acc += i in torch.topk(temp, 1).indices
        top5_acc += i in torch.topk(temp, 5).indices
        top50_acc += i in torch.topk(temp, 50).indices
        rst[i] = temp.numpy()

    print("=========================================")
    print(f"top1 acc: {top1_acc / 804}")
    print(f"top5 acc: {top5_acc / 804}")
    print(f"top50 acc: {top50_acc / 804}")
    print("=========================================")

    # np.save('rst_epoch{}.npy'.format(epoch), rst)

    # print('Test checkpoint epoch {}.'.format(epoch))

    # gen_tsample(50)

    # tsample = np.load('tsample_{}.npy'.format(50))
    # get_top(tsample, rst)


if __name__ == '__main__':
    main()