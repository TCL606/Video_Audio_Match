import numpy as np

if __name__ == '__main__':
    idx = np.arange(3339)
    np.random.seed(5201314)
    np.random.shuffle(idx)
    train = idx[:3000]
    valid = idx[3000:]
    np.save('train.npy', train)
    np.save('valid.npy', valid)
