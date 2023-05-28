import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]


X ,y = create_data()
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#实现三个接口fit, predit, score
class AdaBoost_DesisionTree:
    def __init__(self, n_estimators=50):
        self.clf_num = n_estimators
        self.best_thres = 0
        self.best_fea_id = 0
        self.best_err = 1
        self.alpha = []
        #print(self.M)
        self.best_op = 1
    

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        n = X.shape[1]
        for i in range(n):
            feature = X[:,i]    #选定特征列i
            fea_unique = np.sort(np.unique(feature))
            for j in range(len(fea_unique)-1):
                thres = (fea_unique[j] + fea_unique[j+1]) / 2
                for op in (0,1):
                    #1:表示大于阈值为正，0 表示小于阈值为正
                    y_ = 2*(feature >= thres)-1 if op == 1 else 2*(feature < thres)-1
                    err = np.sum((y_ != y)*sample_weight)
                    if (err < self.best_err):
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self

    def predict(self, X):
        feature = X[:, self.best_fea_id]
        #print(self.best_op)
        return 2*(feature >= self.best_thres)-1 if self.best_op == 1 else 2*(feature < self.best_thres)-1

    def score(self, X, y):
       # print(len(X), len(y))
        y_pre = self.predict(X)
        #print(self.weights)
        #print(y)
        #return np.sum((y_pre == y)*self.weights)
        return np.mean(y_pre == y)


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
    def fit(self, X, y):
        sample_weight = np.ones(len(X)) / len(X)
        for i in range(self.n_estimators):
            dtc = AdaBoost_DesisionTree().fit(X,y,sample_weight)
            print("best feature:",dtc.best_fea_id)
            alpha = 1/2 * np.log((1-dtc.best_err)/dtc.best_err)
            y_pred = dtc.predict(X)
            sample_weight *= np.exp(-alpha*y*y_pred)
            sample_weight /= sum(sample_weight)
            self.estimators.append(dtc)
            self.alphas.append(alpha)
        return self

    def predict(self, X):
        y_pred = np.empty((len(X),self.n_estimators))
        print(y_pred.shape)
        for i in range(self.n_estimators):
            y_pred[:,i] = self.estimators[i].predict(X)
        y_pred = y_pred * np.array(self.alphas)
        return 2*(np.sum(y_pred, axis=1)>0)-1

    def score(self,X,y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
#加载训练数据
X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
              [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])

clf = AdaBoost().fit(X,y)


#print(len(X_test), len(y_test))
print("my:",clf.score(X, y))

dlf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
dlf.fit(X,y)
print(dlf.score(X, y))

