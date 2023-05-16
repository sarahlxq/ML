import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

arr = pd.read_csv("/home/lixq/work/new4.csv", header=None, delimiter=",", dtype=float)
sns.set()
plt.figure(figsize=(30,18))
ax = sns.heatmap(arr, vmin=240, vmax=243, cmap="Blues")
plt.xlim(0, 70)
plt.ylim(0, 103)
plt.show()
plt.savefig('map.png')
