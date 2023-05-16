"""
逻辑斯蒂回归，随机梯度下降法
"""
import numpy as np
import time
import random
from itertools import islice
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
"""
读取训练数据
"""
def loadData(file):
    fr = open(file,'r')
    data = []
    label = []
    for line in islice(fr,1,None):
        splited = line.strip().split(',')
        data.append([int(num)/255 for num in splited[1:]])
        a = int(splited[0])
        if a > 4:
            label.append(1)
        else:
            label.append(0)

    SampleNum = len(data)
    for i in range(SampleNum):
        data[i].append(1)
            
    # 返回数据的特征部分和标记部分
    return data, label

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length','sepal width','petal length','petal width','label']
    data = np.array(df.iloc[:100,[0,1,-1]])
    #print(data[:,:2])
    return data[:,:2], data[:,-1]



def predictVal(x, w):
    exp_wi = np.exp(np.dot(w,x))
    p = exp_wi / (1+exp_wi)
    if p > 0.5:
        return 1
    return 0

def data_matrix(X):
    data_mat = []
    for d in X:
        data_mat.append([1.0, *d])
    return data_mat

def Logistic(dataSet, label, itertime):
    print(np.shape(dataSet))
    featureNum = np.shape(dataSet)[1]
    
    #data_mat = data_matrix(dataSet)
    w = np.zeros(featureNum)
    print(w)
    #print(w)
    h = 0.001
    sampleNum = len(dataSet)
    dataSet = np.array(dataSet)
    for i in range(itertime):
        a = 0
        cnt = 0
        while a == 0 or cnt < 10:
            s = random.sample(range(0,sampleNum-1),1)[0]
            #print(s)
            xi = dataSet[s]
            yi = label[s]
            if predictVal(xi,w) != yi:
                exp_wi = np.exp(np.dot(w,xi))
                w += h*(xi*yi-(xi*exp_wi)/(1+exp_wi))
                a = 1
            else:
                cnt += 1
            #print("cnt: ",cnt)

    return w

"""
模型测试
输入:data测试集,label 标记,优化的W
输出:Acc正确率
"""
def Classifier(data, label,w):
   # print((data))
    sampleNum = len(data)
    errorCnt = 0
    for i in range(sampleNum):
        result = predictVal(w, data[i])
        if result != label[i]:
            errorCnt += 1
    Acc = 1- errorCnt/ sampleNum
    return Acc

if __name__ == "__main__":
    print('start loading')

    X,y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #print(X_train)
    print('start trainning')
    b = np.ones(len(X_train))
    X_train = np.insert(X_train,2,values=b,axis=1)
    X_test = np.hstack((X_test, np.ones((len(X_test), 1))))
   # X_test = np.insert(X_test,2,values=b,axis=1)
    #print(X_train)
    start = time.time()
    w = Logistic(X_train, y_train, 200)
    print(w)
    print('end trainning')
    end = time.time()
    print("trainning time: ",end-start)
    acc = Classifier(X_test, y_test,w)
    print("acc: ",acc)
    x_ponits = np.arange(4, 8)
    y_ = -(w[0]*x_ponits + w[2])/w[1]
    plt.plot(x_ponits, y_)

    #show_graph()
    plt.scatter(X[:50,0],X[:50,1], label='0')
    plt.scatter(X[50:,0],X[50:,1], label='1')
    plt.legend()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    clf.score(X_test,y_test)
    print(clf.coef_, clf.intercept_)
    y_ = -(clf.coef_[0][0]*x_ponits+clf.intercept_)/clf.coef_[0][1]
    #plt.plot(x_ponits,y_)
    #plt.scatter(X[:50,0],X[:50,1], label='0')
    #plt.scatter(X[50:,0],X[50:,1], label='1')
    #plt.legend()
   # plt.show()

