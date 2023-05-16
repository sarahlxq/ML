import math
from itertools import combinations

def L(x,y,p=2):
    if (len(x) == len(y)) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0

x1 = [1,1]
x2 = [5,1]
x3 = [4,4]
for i in range(1,5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2,x3]}
   # print(min(zip(r.values(), r.keys())))
#L(x1,x2)
#L(x1,x3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train)
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X-self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
        #print("knn_list", knn_list)

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            #print("max_index", max_index)
            dist = np.linalg.norm(X-self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        print("knn_list", knn_list)
        knn = [k[-1] for k in knn_list]
        print(knn)
        count_pairs = Counter(knn)
        print("count_pairs:",count_pairs)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        print("max_count:",max_count)
        return max_count
    def score(self,X_test, y_test):
        right_count = 0
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train, y_train)
#print("score:",clf.score(X_test, y_test))
test_point = [6.0, 3.0]
#print('Test Point: {}'.format(clf.predict(test_point)))

from sklearn.neighbors import KNeighborsClassifier
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
clf_sk.score(X_test,y_test)

from math import sqrt
from collections import namedtuple
result = namedtuple("Result_tuple","nearst_point  nearst_dist  nodes_visited")

class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt   #节点数据data_set[split_pos]
        self.split = split
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self, data):
        self.near_dist = np.inf
        self.nearest_node = None
        k = len(data[0])
        def CreateNode(split, data_set):
            if not data_set:
                return None

            #print("split:",split)
            #print("dataset:",data_set)
            data_set.sort(key = lambda x:x[split])
            split_pos = len(data_set) // 2
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KdNode(
                median, split, CreateNode(split_next,data_set[:split_pos]),
                CreateNode(split_next, data_set[split_pos+1:])
            )
        self.root = CreateNode(0,data)
    
    def find_nearst(self, tree, point):
        k = len(point)
        def visit(kd_node):
            #print("visit",kd_node.dom_elt)
            if kd_node != None:
                sp = kd_node.split
                data = kd_node.dom_elt
                #print("node:",kd_node.dom_elt)
                dis = (point[sp] - data[sp])
                #print("dis:{}".format(point[sp]-data[sp]))
                visit(kd_node.left if dis < 0 else kd_node.right)
                #print("visit:",kd_node.dom_elt)
                curr_dist = np.linalg.norm(kd_node.dom_elt-point,2)
                #print("dist:",curr_dist)
                if curr_dist < self.near_dist:
                    self.near_dist = curr_dist
                    self.near_node = kd_node.dom_elt
                    print("near_dist, near_node",self.near_dist, self.near_node)
                #print("return dist",dis)
                if abs(dis) < self.near_dist:
                    visit(kd_node.right if dis < 0 else kd_node.left)
        visit(tree.root)
        return self.near_node, self.near_dist  


def preorder(root):
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]

from time import clock
from random import random

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]
 
# 产生n个k维随机向量 
def random_points(k, n):
    return [random_point(k) for _ in range(n)]  

N = 4000000
t0 = clock()
kd2 = KdTree(random_points(3,N))
ret2 = kd2.find_nearst(kd2, np.array([0.1, 0.5, 0.8]))
t1 = clock()
print("time: ",t1-t0,"s")
print(ret2)
#kd = KdTree(data)
#preorder(kd.root)
#print(kd.root)
#print(kd.find_nearst(kd, np.array([2,4.5])))