import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
#print(df['label'])
df.columns = ['sepal length','sepal width','petal length','petal width','label']
#print(df.label.value_counts())
df.label.value_counts()
"""

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
"""
#plt.show()
data = np.array(df.iloc[:100,[0,1,-1]])
#print(data.shape)
X,y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


class Model:
    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        #print(data[0])

    def sign(self, x,w,b):
        y = np.dot(w,x)+b
        #print(y)
        return y
    
    def fit(self, X_train, y_train):
        isWrong = False
        while not isWrong:
            wrong_count = 0
            for d in range(len(X_train)):
                #print("y_: ",y_train[d])
                #print(self.w)
                if y_train[d] * self.sign(X_train[d],self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate*y_train[d]*X_train[d]
                    self.b = self.b + self.l_rate*y_train[d]
                    wrong_count += 1
            if wrong_count == 0:
                isWrong = True
        return 'Perceptron Model!'

perceptron = Model()
perceptron.fit(X,y)

x_points = np.linspace(4,7,10)
y_ = -(perceptron.w[0]*x_points+perceptron.b)/perceptron.w[1]
plt.plot(x_points,y_)
plt.plot(data[:50,0],data[:50,1],'bo',color='blue',label='0')
plt.plot(data[50:100,0], data[50:100,1],'bo',color='orange',label='1')
plt.xlabel('sepal width')
plt.ylabel('sepal width')
plt.legend()
plt.show()