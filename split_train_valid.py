import numpy as np

if __name__ == '__main__':
    # idx = np.arange(3339)
    # np.random.seed(5201314)
    # np.random.shuffle(idx)
    # train = idx[:3000]
    # valid = idx[3000:]
    # np.save('train.npy', train)
    # np.save('valid.npy', valid)

    test_top1 = np.load('/mnt/e/清华/大三秋/视听信息系统导论/大作业/STD2022/output/test/test_top1.npy').squeeze(1)
    print(test_top1)
