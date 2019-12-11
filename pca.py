from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

rng = np.random.RandomState(3)
data = np.dot(rng.rand(2, 2), rng.randn(2, 600)).T
plt.scatter(data[:, 0], data[:, 1], c = 'green')
#plt.axis('equal');
M = mean(data.T, axis=1)
C = data - M
V = cov(C.T)
values, vectors = eig(V)
print(values)
print('eigen vector (using sklearn)')
print(vectors)
P = vectors.T.dot(C.T)
print('------------------------------------------------------')

def hebbian_pca(data, number_of_components =2, w =[], lr =0.01, iterations =200):
    npc = number_of_components
    if np.array(w).shape[0] > 0: pass # do better checking 
    else: w = np.random.uniform(-1,1,(data.shape[1],npc))
    for iters in range(iterations):
        for x in data:
            y = np.dot(w.T,x[:,True])
            Ymat= []
            for i in range(npc):
                Ymat += [y]
            Y_mat = np.array(Ymat)[:,:,0]
            tril_mat = np.tril(Y_mat,k=0).T
            S_mat = np.dot(w,tril_mat)
            Xmat= []
            for i in range(npc):
                Xmat += [x]
            X_mat = np.array(Xmat).T
            dw = lr * y[:,0]* (X_mat - S_mat)
            w += dw
    return w

data -= data.mean(0)
w = hebbian_pca(data,number_of_components= 2)
print('eigen vector (using Neural Net)')
print(w)
P = w.T.dot(C.T)
plt.scatter(P.T[:, 0], P.T[:, 1], c = 'red')
plt.show()

