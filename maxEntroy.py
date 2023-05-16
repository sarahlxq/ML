import math
from copy import deepcopy

dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]

class MaxEntropy:
    def __init__(self, EPS=0.005):
        self._samples = []
        self._Y = set()
        self._numXY = {}
        self._N = 0
        self._Ep_ = []
        self._xyID = {}
        self._n = 0
        self._C = 0
        self._IDxy = {}
        self._w = []
        self._EPS = EPS
        self._lastw = []
    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
    #取dataset的行为items
        for items in self._samples:
    #y为第一列，X为2至最后一列
            y = items[0]
            X = items[1:]
            #print("items ",X)
            self._Y.add(y)
            for x in X:
                if (x,y) in self._numXY:
                    self._numXY[(x,y)] += 1
                else:
                    self._numXY[(x,y)] = 1
            #print("(x,y) ",self._numXY)
        #print(len(self._numXY))   #表示特征fi的组合
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max(len(sample)-1 for sample in self._samples)
        self._w = [0]*self._n
        self._lastw = self._w[:]
        print("N: n, C",self._N,self._n,self._C)
# 计算特征函数fi 的经验分布期望
        self._Ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):
            self._Ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy
# 计算每个Zx
    def _Zx(self, X):
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
                    print("(x,y)",(x,y),"self ID",self._xyID[(x,y)],"w ",self._w[self._xyID[(x,y)]])
            zx += math.exp(ss)
            print("zx,ss,exp:",zx, ss,math.exp(ss))

        return zx   

    def _model_pyx(self, y, X):
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x,y) in self._numXY:
                ss += self._xyID[(x,y)]
        pyx = math.exp(ss) / zx
        print("ss:,zx,pyx",ss,zx,pyx)
        return pyx
#计算模型的期望
    def _model_ep(self, index):
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
            #print("ep,pyx ",ep,pyx)
        return ep

    def _convergence(self):
        for last, now in zip(self._lastw, self._w):
            if abs(last-now) >= self._EPS:
                return False
        return True
    
    def predict(self, X):
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x,y) in self._numXY:
                    ss += self._w[self._xyID[(x,y)]]
            if Z != 0:
                pyx = math.exp(ss) / Z
                result[y] = pyx
        return result

    def train(self, maxiter=10):
        for loop in range(maxiter):
            print("iter:%d" % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)
               # print("self._Ep_[i]:",self._Ep_[i],"ep: ",ep)
                if self._Ep_[i] / ep > 0:
                    self._w[i] += math.log(self._Ep_[i] / ep) / self._C  # 更新参数
            if self._convergence():
                break



maxent = MaxEntropy()
x = ['overcast', 'mild', 'high', 'FALSE']
maxent.loadData(dataset)
#print("maxent: ",maxent._samples[0])
maxent.train()
print('predict:', maxent.predict(x))



