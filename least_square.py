import numpy as np
import scipy as sp
from scipy.optimize import leastsq
#import matplotlib as plt
import matplotlib.pyplot as plt

#print(np.poly1d([1,2,3]))
def real_func(x):
    y = np.sin(2*np.pi*x)
    return y

def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)
def residuals_func(p,x,y):
   # print(len(p))
    ret = fit_func(p,x) - y
    return ret


x = np.linspace(0,1,10)
x_points = np.linspace(0,1,1000)
y_ = real_func(x)
y = [np.random.normal(0,0.1) + y1 for y1 in y_]
#print(np.random.normal(0,0.1))

regularization = 0.0001
def residuals_func_regularuization(p,x,y):
    #print(p)
    ret = fit_func(p,x) - y
   # print("before",ret)
    ret = np.append(ret,np.sqrt(0.5*regularization*np.square(p)))
  #  print("after",ret)
    return ret

def fitting(M=0):
    p_init = np.random.rand(M+1)
    p_lsq = leastsq(residuals_func, p_init, args=(x,y))
    p_lsq_regularization = leastsq(residuals_func_regularuization, p_init, args=(x,y))
    print('Fitting Parameters:', p_lsq_regularization[0])

    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0],x_points), label='fitting curve')
    plt.plot(x_points, fit_func(p_lsq_regularization[0],x_points), label='regularization')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()
    plt.show()
    return p_lsq

p_lsq_0 = fitting(M=9)


