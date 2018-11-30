# 并查集

## 介绍
>并查集是一种树形结构，用于处理一些不相交集合的合并及查询问题。它主要涉及两种基本操作：合并和查找。这说明，初始时并查集中的元素是不相交的，经过一系列的基本操作(Union)，最终合并成一个大的集合。而在某次合并之后，有一种合理的需求：某两个元素是否已经处在同一个集合中了？因此就需要Find操作。  
>并查集的优点在于它的时间复杂度，可以达到常数级别的时间复杂度。~O(1)

    
## 实现
### 存储结构
> 并查集是由一个数组组成，它的数组值表示当前index节点的父节点。  
例如：s[5] = 4 表示5号节点的父节点是4号节点
同时它还包含两个操作，分别是union和find操作：  
find函数有一个参数，是为了找到当前参数表示节点的父节点。  
union函数有两个参数，是合并操作，所进行的是把一个节点合并到另一个节点上。  
### 代码实现
> v1 
```
class union_find:
    def __init__(self, n):
        #为了简单起见，通常初始数组的父节点会指向自己。指向-1也可以
        self.father = [i for i in range(n)]

    def find(self, p):
        #通过递归去查找第p个节点对应的父节点
        if self.father[p] == p:
            return p
        return self.find(self.father[p])

    def union(self, p, q):
        #合并p和q对应的集合。使得p的节点成为新的父节点。
        a = self.find(p)
        b = self.find(q)
        if a != b:
            self.father[b] = a 

```
但是，这个find操作需要消耗O(n)的时间复杂度，不是我们所想要的。所以我们要通过`路径压缩`的方法
> v2: 路径压缩  
路径压缩是指在find函数中加入回溯，使得每一个点都的father数组都指向最终的父节点。这样可以做到查找效率是O(1)，实际上是从一个链式的存储变成了一个树状的存储结构。但是这样做可能会丢掉每一个节点的直接父节点，我们并查集中不关心这样的信息，这属于冗余信息，所以我们可以忽略。我们只需要完成find和union两个操作，所以优化的代码为：  
```
class union_find:
    def __init__(self, n):
        self.father = [i for i in range(n)]

    def find(self, p):
        #这里的father都指向了最终的父节点
        if self.father[p] == p:
            return p
        self.father[p] = self.find(self.father[p])
        return self.father[p]

    def union(self, p, q):
        a = self.find(p)
        print(a)
        b = self.find(q)
        print(b)
        if a != b:
            self.father[b] = a 
```
