import numpy as np

if __name__ == '__main__':
    idx = np.arange(3339)
    np.random.seed(5201314)
    np.random.shuffle(idx)
    train_sam = 3229
    train = idx[:train_sam]
    valid = idx[train_sam:]
    np.save('train.npy', train)
    np.save('valid.npy', valid)

    # test_top1 = np.load('/root/bqqi/changli/STD2022/test_top1.npy').squeeze(1)
    # print(test_top1)
