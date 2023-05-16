import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets._samples_generator import make_blobs


#print("hello py")
a = np.array([[1,4],[2,5],[3,6]])
#print(a.T)
cov = np.cov(a.T)
#print(cov)
eigval, eigvect = (np.linalg.eig(cov))   #返回特征值和特征向量
aver = np.mean(a, axis=0)
#print(aver)
b = []
#print(len(a[0]))
for i in range(len(a[0])):
    b.append(a[:,i] - aver[i])
b = np.asarray(b)
res = b.T*eigvect[0]

#生成数据集
X,y = make_blobs(n_samples=10000,n_features=3, centers=[[3,3,3000],[0,0,0],[1,1,1000],[2,2,2]],
                 cluster_std=[0.2,0.1,0.2,0.2], random_state=8)

fig = plt.figure()
ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)
#ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(X[:, 0], X[:,1], X[:,2], marker='o')

cov = np.cov(X.T)
eigval, eigvect = (np.linalg.eigh(cov))
print(eigval)
print(eigvect)

a = np.hstack((eigvect[:,-1].reshape(3,-1),eigvect[:,1].reshape(3,-1)))
#print(a)
X = X - X.mean(axis=0)
X_new = X.dot(a)
#print(X_new)
fig2 = plt.figure()
plt.scatter(X_new[:,0], X_new[:,1], marker='o')
#plt.show()
print("**************SVD********************")
#使用SVD 方法
fig3 = plt.figure()
from sklearn.decomposition import PCA
pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_)
print(pca.components_)
X_new2 = pca.transform(X)
print(X_new2)
#plt.scatter(X_new2[:,0], X_new2[:,1], marker='o')
plt.scatter(X_new2[:,0],y, marker='o')
plt.show()