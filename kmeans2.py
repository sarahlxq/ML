from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data

def distEclud(vecA, vecB):
    return sum(pow(vecA - vecB,2))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = np.zeros((k,n))
    #print("1",shape(centroids))
    #centroids = mat(zeros((k,n))) 
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1).T
        #print(centroids[:,j])
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = np.zeros((m,2))    #保存了每个样本与质心的最小距离
    centroids = createCent(dataSet, k)
    #print(centroids)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                #print("centroids:",centroids[j,:])
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
              #  print(clusterAssment[i,0])
                clusterAssment[i,:] = minIndex, minDist
        #print(centroids)
        color_dict = {0:"blue", 1:"red", 2:"black", 3:"yellow"}
        plt.figure(2)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0] == cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
            #print(ptsInClust)
            plt.scatter(ptsInClust[:,0], ptsInClust[:,1], color = color_dict[cent])
       # print(centroids)
        plt.scatter(centroids[:,0], centroids[:,1], marker='^',color="green")
       # plt.show()
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = np.zeros((m,2))
    centroid0 = np.mean(dataSet, axis=0)   #对每个特征的样本求平均
    centList = [centroid0]
    while(len(centList) < k):
        lowestSSE = np.inf
        for j in range(len(centList)):
            print(len(centList))
            ptsIncurrCluster = dataSet[np.nonzero(clusterAssment[:,0] == j)[0],:]
            #print(np.nonzero(clusterAssment[:,0] == j)[0])
            centroid, splitAss = kMeans(ptsIncurrCluster,2,distMeas)
            print("centroid",centroid)
            splitSse = sum(splitAss[:,1])
            splitNotSse = sum(clusterAssment[np.nonzero(clusterAssment[:,0] != j)[0],1])
            #print("ptsIncurrCluster",ptsIncurrCluster)
            print("size",shape(ptsIncurrCluster))
            print("splitSse",splitSse)
            print("splitNotSse",splitNotSse)
            
            if (splitSse+splitNotSse) < lowestSSE:
                lowestSSE = splitSse+splitNotSse
                bestCentToSplit = j
                bestNewCents = centroid
                bestClustAss = splitAss.copy()
              #  print("newcent",bestNewCents[bestCentToSplit,:])
            #print("1 sbestAS",bestClustAss)
        bestClustAss[np.nonzero(bestClustAss[:,0] == 1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0] == 0)[0],0] = bestCentToSplit
        #print("2 sbestAS",bestClustAss)
        print("the bestCentToSplit is: ",bestCentToSplit)
        print("the len of bestClustAss is: ",len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        print(centList)
        clusterAssment[np.nonzero(clusterAssment[:,0] == bestCentToSplit)[0],:] = bestClustAss
    return centList, clusterAssment


        
    #print(type(centList))


fileName = "datatest/testSet.txt"
data = loadDataSet(fileName)

#plt.figure(1)
#plt.scatter(data[:,0],data[:,1], color="green")
#plt.show()
#print(distEclud(data[:,0], data[:,1]))

#print(randCent(data, 1))
#result, clusterAssment = kMeans(data, 4)
#print(clusterAssment)
myCentroids, clustAssing = biKmeans(data, 4)
centroids = np.array([i for i in myCentroids])
#print(centroids)
y_kmeans = clustAssing[:, 0]
#print(y_kmeans)
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], s=50)
plt.title("未聚类前的数据分布")
plt.subplots_adjust(wspace=0.5)
plt.subplot(122)
plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.5)
plt.title("用二分K-Means算法原理聚类的效果")
plt.show()
