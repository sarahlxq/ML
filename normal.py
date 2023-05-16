import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(color_codes=True)
train_data = pd.read_csv('/home/lixq/work/leaning_test/zhengqi_train.txt',sep='\t',encoding='utf-8')
test_data = pd.read_csv('/home/lixq/work/leaning_test/zhengqi_test.txt', sep='\t', encoding='utf-8')
#print(train_data)
#print(train_data.values)
#print(train_data.head())
#train_data.plot.box(train_data)
#plt.grid(linestyle="--", alpha=0.3)
#plt.show()
#sns.displot(train_data("v0"))
#print(train_data[["V0"]])

"""

rows, cols = train_data.shape
plt.figure(figsize=(4*6,4*len(test_data.columns)))
ax = plt.subplot(10, 2,1)
ax = sns.kdeplot(train_data["V0"],fill=True, color="b")
#ax = sns.kdeplot(test_data[col],fill=True, color='r')
ax = ax.legend(["train","test"])
plt.show()
"""

dist_cols = 6
rows, cols = test_data.shape
train_rows = len(train_data.columns)
#plt.figure(figsize=(4*dist_cols,4*train_rows))
print(rows, cols, train_rows)
i = 0

for col in test_data.columns:
    i += 1
    #i = (i % 24) + 1
    ax = plt.subplot(train_rows, dist_cols,i)
    ax = sns.kdeplot(train_data[col],fill=True, color="b")
    ax = sns.kdeplot(test_data[col],fill=True, color='r')
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train","test"])
    break

plt.tight_layout()
plt.show()

