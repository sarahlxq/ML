from math import log

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']             #分类属性
    return dataSet, labels                #返回数据集和分类属性

def calEntropy(dataSet):
    totalNum = len(dataSet)
    labelCnts = {}
    for vecFeature in dataSet:
        curLabel = vecFeature[-1]
        if curLabel not in labelCnts.keys():
            labelCnts[curLabel] = 0
        labelCnts[curLabel] += 1
    entropy = 0.0
    for key in labelCnts:
        prob = float(labelCnts[key]) / totalNum
        entropy += (-prob * log(prob,2))
    return entropy

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reduceVec = featureVec[:axis]
            #print(reduceVec)
            reduceVec.extend(featureVec[axis+1:])
            retDataSet.append(reduceVec)
    return retDataSet
 
def calBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #获取dataSet中的行放在example中，然后取example的第i列
        featureList = [example[i] for example in dataSet]  
        uniqueVals = set(featureList)
        #print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            #print(subDataSet)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为:%.3f" %(i,infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
创建决策树
递归的停止条件：
1.所有类标签完全相同
2.用完了所有的特征，仍然不能将数据集划分成仅包含唯一类别的分组

"""

if __name__ == '__main__':
    dataSet, features = createDataSet()
    #print(dataSet)
    print(calEntropy(dataSet))
    print(calBestFeatureToSplit(dataSet))
   # print()
    #print(splitDataSet(dataSet, 0, 0))
    #print(calcShannonEnt(dataSet))