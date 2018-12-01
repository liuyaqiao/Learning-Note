'''
591. 连接图 III
给一个图中的 n 个节点, 记为 1 到 n . 在开始的时候图中没有边.
你需要完成下面两个方法:

connect(a, b), 添加一条连接节点 a, b的边
query(), 返回图中联通区域个数
样例
5 // n = 5
query() 返回 5
connect(1, 2)
query() 返回 4
connect(2, 4)
query() 返回 3
connect(1, 4)
query() 返回 3
'''
class ConnectingGraph3:
    """
    @param a: An integer
    @param b: An integer
    @return: nothing
    """
    def __init__(self, n):
        self.father = [i for i in range(n + 1)]
        self.count = n
        
        
    def find(self, x):
        if self.father[x] == x:
            return x
        self.father[x] = self.find(self.father[x])
        return self.father[x]
        
        
    def connect(self, a, b):
        # write your code here
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.count -= 1
    """
    @return: An integer
    """
    def query(self):
        # write your code here
        return self.count