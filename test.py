import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T

ax = sns.kdeplot(x, fill=True, color="r")

#plt.show()

import torch
x = torch.rand(5,3)
print(x)
import torch
torch.cuda.is_available()

