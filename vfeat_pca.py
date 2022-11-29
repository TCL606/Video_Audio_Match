import numpy as np
import os
from sklearn.decomposition import PCA

pca = PCA(n_components=512)

vpath = 'Train/vfeat2048'
spath1 = 'Train/vfeat'
spath2 = 'Test/Clean/vfeat'
spath3 = 'Test/Noise/vfeat'
vf = os.listdir(vpath)
l = len(vf)
print(l)
X = []
Xt = []
for i in range(l):
    feat = np.load(os.path.join(vpath, '%04d.npy'%i))
    print(i)
    X.append(feat)
vpath = 'Test/Clean/vfeat2048'
vf = os.listdir(vpath)
l0 = l
l = len(vf)
print(l)
for i in range(l):
    print(i)
    feat = np.load(os.path.join(vpath, '%04d.npy'%i))
    X.append(feat)
vpath = 'Test/Noise/vfeat2048'
vf = os.listdir(vpath)
print(len(vf))
for i in range(len(vf)):
    print(i)
    feat = np.load(os.path.join(vpath, '%04d.npy'%i))
    Xt.append(feat)

X = np.concatenate(X, axis=0)
Xt = np.concatenate(Xt, axis=0)
print(X.shape)
print(Xt.shape)
newX = pca.fit_transform(X)
newXt = pca.transform(Xt)

assert newX.shape[0] == (l0 + l) * 10

for i in range(l0):
    feat = newX[i*10: i*10 + 10]
    np.save(os.path.join(spath1, '%04d.npy'%i), feat)

for i in range(l):
    feat = newX[(i+l0)*10:(i+l0+1)*10]
    np.save(os.path.join(spath2, '%04d.npy'%i), feat)

for i in range(l):
    feat = newXt[i*10:(i+1)*10]
    np.save(os.path.join(spath3, '%04d.npy'%i), feat)


