from sklearn.linear_model import RidgeCV

ridgecv = RidgeCV(alphas=[0.01,0.1,0.5,1,3,5,7,10,20,100])
ridgecv.fit()
ridgecv.alpha_
def ridge_regression(X, y, lamba=0.2):
    XTX = X.T*X
    