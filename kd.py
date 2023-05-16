import numpy as np

class Node:
    def __init__(self,data,sp=0,left=None,right=None):
        self.data = data
        self.sp = sp  #0是按特征1排序，1是按特征2排序
        self.left = left
        self.right = right
        
    def __lt__(self, other):
        return self.data < other.data

class KDTree:
    def __init__(self,data):
        self.dim = data.shape[1]
        self.root = self.createTree(data,0)
        self.nearest_node = None
        self.nearest_dist = np.inf #设置无穷大

    def createTree(self,dataset,sp):
        if len(dataset) == 0:
            return None

        dataset_sorted = dataset[np.argsort(dataset[:,sp])] #按特征列进行排序
        #获取中位数索引
        mid = len(dataset) // 2
        #生成节点
        left = self.createTree(dataset_sorted[:mid],(sp+1)%self.dim)
        right = self.createTree(dataset_sorted[mid+1:],(sp+1)%self.dim)
        parentNode = Node(dataset_sorted[mid],sp,left,right)
       
        return parentNode
    
    def nearest(self, x):
        def visit(node):
            if node != None:
                dis = node.data[node.sp] - x[node.sp]
                print("node, dis",node.data, dis)
                #访问子节点
                visit(node.left if dis > 0 else node.right)
                #查看当前节点到目标节点的距离 二范数求距离
                curr_dis = np.linalg.norm(x-node.data,2)
                print("node, curr_dis",node.data, curr_dis)
                #更新节点
                if curr_dis < self.nearest_dist:
                    self.nearest_dist = curr_dis
                    self.nearest_node = node
                #比较目标节点到当前节点距离是否超过当前超平面，超过了就需要到另一个子树中
                if self.nearest_dist > abs(dis): #要到另一面查找 所以判断条件与上面相反
                    print("nearest_dist, dist,node.data ",self.nearest_dist,dis, node.data)
                    visit(node.left if dis < 0 else node.right)
        
        #从根节点开始查找
        node = self.root
        visit(node)
        return self.nearest_node.data,self.nearest_dist

data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
kdtree = KDTree(data)  #创建KDTree
node,dist = kdtree.nearest(np.array([2,4.5]))
print(node,dist)