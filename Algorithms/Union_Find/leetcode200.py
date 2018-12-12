'''
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1
Example 2:

Input:
11000
11000
00100
00011

Output: 3
'''

常见的BFS和DFS算法见
[leetcode200](https://github.com/liuyaqiao/Algorithms/blob/master/src/200.py)

```
class Solution():   
    father = []
       
    def find(self, x):
        if x == self.father[x]:
            return x
        self.father[x] = self.find(self.father[x])
        return self.father[x]
            
    def union(self, p, q):
        a = self.find(p)
        b = self.find(q)
        if a != b:
            self.father[b] = a
    
    def numIslands(self, grid):
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        
        rows = len(grid)
        cols = len(grid[0])
        self.father = [i for i in range(rows * cols)]

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    if i > 0 and grid[i - 1][j] == '1':
                        self.union(i * cols + j, (i - 1) * cols + j)
                    if i < rows - 1 and grid[i + 1][j] == '1':
                        self.union(i * cols + j, (i + 1) * cols + j)
                    if j > 0 and grid[i][j - 1] == '1':
                        self.union(i * cols + j, i * cols + j - 1)
                    if j < cols - 1 and grid[i][j + 1] == '1':
                        self.union(i * cols + j, i * cols + j + 1)
                else:
                    k = i * cols + j
                    self.father[k] = -1
        count = 0
	#需要注意的是i的取值范围是rows - 1#
	#对k的i*cols+j的取值#
	#筛选的时候通过i和father[i]的值来判断#

        for i in range(cols * rows):
            if self.father[i] == i:
                count += 1
        return count
    #筛选的时候要注意#
```
其中有几个要注意的地方：
1.要注意一下对row和col的取值范围，容易造成数组的越界。
2.判断有几个连通区域的时候，使用father[i] 和 i比较的方法来决定
3.这里可以使用一个简单的循环来遍历整个图。具体见下面的代码：

