import numpy as np
import pandas as pd
import scipy.stats as sps

class linear_model:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.b = np.linalg.solve(x.T@x, x.T@y)
        self.e = y - x@self.b
        self.vb = self.e.var() * np.linalg.inv(x.T@x)
        self.se = np.sqrt(np.diagonal(self.vb))
        self.t = self.b / self.se
        self.p = 2 * sps.norm.cdf(-np.abs(self.t))
        self.rsq = 1 - (self.e.var() / y.var())
    def newey_west_estimator(self):
        x = self.x
        e = self.e
        w = np.zeros((e.size, e.size))
        sigma = 0
        for i in range(1, e.size):
            sigma += e[i] * e[i-1]
        sigma_1 = sigma / e.size
        sigma_sq = np.sum(e) / e.size
        np.fill_diagonal(w, sigma_sq)
        for i in range(e.count()-1):
            w[i][i+1] = sigma_1
            w[i+1][i] = sigma_1
        return np.linalg.inv(x.T@x)@(x.T@w@x)@np.linalg.inv(x.T@x)

df = pd.read_csv('BWGHT.csv')
df['(intercept)'] = 1
x = df[['(intercept)','cigs','faminc']]
y = df['bwght']

lm = linear_model(x, y)
print(lm.newey_west_estimator())