import numpy as np

#加载训练数据
X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
              [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])

class AdaBoost:
    def __init__(self, X,y,tol=0.05, max_iter=10):
        self.X = X
        self.y = y
        self.tol = tol
        self.max_iter = max_iter
        self.w = np.full((X.shape[0]), 1 / X.shape[0])
        self.G = []

    
    def build_stump(self):
        m, n = np.shape(self.X)
        e_min = np.inf
        sign = None
        best_stump = {}
        for i in range(n):
            pass

    def base_estimator(X, dimen, threshVal, threshIneq):
        ret_array = np.ones(np.shape(X)[0])
        if threshIneq == 'lt'

    def update_w(self, alpha, predict):
        P = self.w * np.exp(-alpha * self.y * predict)
        self.w =  P / P.sum()


    def fit(self):
        G = 0
        for i in range(self.max_iter):
            best_stump, sign, error = self.build_stump()
            alpha = 1/2 * np.log((1-error) / error)
            best_stump['alpha'] = alpha
            self.G.append(best_stump)
            G += alpha * sign
            y_predict = np.sign(G)
            error_rate = np.sum(np.abs(y_predict-self.y)) / 2 / self.y.shape[0]
            if error_rate < self.tol:
                print("迭代次数:",i+1)
                break
            else:
                self.update_w(alpha, y_predict)


    def predict(self, X):
        m = np.shape(X)[0]
        G = np.zeros(m)
