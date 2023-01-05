from argparse import ArgumentParser
from sklearn.cluster import KMeans
import numpy as np
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--clusters', type=int, default=15)
    parser.add_argument('--train_idx', type=str, default='./data/train_3005.npy')
    parser.add_argument('--data_dir', type=str, default='../Train/afeat')
    parser.add_argument('--output_dir', type=str, default='./data/kmeans')
    args = parser.parse_args()
    return args

def load_data(args):
    # Load the data
    X_idx = np.load(args.train_idx)
    path = os.path.join(args.data_dir)
    X_train = np.zeros([len(X_idx), 1280])
    for i in range(len(X_idx)):
        X_train[i] = np.load(os.path.join(path, '%04d.npy' % X_idx[i])).reshape(1280)
    return X_train

def cluster(x, args):
    cluster = KMeans(n_clusters=args.clusters)
    y_pred = cluster.fit(x).labels_
    return y_pred


def main():
    args = parse_args()

    # Load the data
    X_train = load_data(args=args)
    y_pred = cluster(X_train, args=args)
    if not os.path.exists(args.output_dir):
        os.system(f'mkdir {args.output_dir}')
    np.save(os.path.join(args.output_dir, f'kmeans_3005_{args.clusters}.npy'), y_pred)

if __name__ == '__main__':
    main()