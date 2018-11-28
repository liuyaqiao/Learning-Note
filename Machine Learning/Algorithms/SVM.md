# Support Vector Machine
>简介：
>定义在特征空间上的间隔最大的分类器。
## 出发点和思路
我们在空间中可以找到无数条线做分类，都可以达到分类的效果。但是分类器有好坏之分，我们的出发点是选择一个能起到最好的分类效果的分类器。这就是我们的出发点。从直观上看，我们应该寻找在所有样本最中间的一条直线做为最终的分类器，这样的分类器对噪音容忍度比较高，对未知数据的泛化能力比较强。  
## 间隔公式
在样本空间，划分超平面可以通过如下的线性方程来描述：  
<a href="https://www.codecogs.com/eqnedit.php?latex=w^{T}x&space;&plus;&space;b&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{T}x&space;&plus;&space;b&space;=&space;0" title="w^{T}x + b = 0" /></a>,  
其中，w表示法向量决定了平面的方向，b是位移量也叫偏执量，决定了超平面和原地之间的距离。空间中任意一点在这个超平民的距离可以表示为：
<a href="https://www.codecogs.com/eqnedit.php?latex=r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" title="r = \frac{\left | w^{T} x + b \right |}{\left \| w \right \|}" /></a>  
![SVM](https://github.com/liuyaqiao/Learning-Note/blob/master/svm.png)

>这里的公式可以通过一个线段到平面的投影来证明。




几何间隔：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;_{i}&space;=&space;y_{i}(wx_{i}&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;_{i}&space;=&space;y_{i}(wx_{i}&space;&plus;&space;b)" title="\gamma _{i} = y_{i}(wx_{i} + b)" /></a>  
函数间隔：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_{i}&space;=&space;y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|&space;}&space;x_{i}&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_{i}&space;=&space;y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|&space;}&space;x_{i}&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})" title="\gamma_{i} = y_{i}(\frac{w}{\left \| w \right \| } x_{i} + \frac{b}{\left \| w \right \|})" /></a>  
## 构造的凸优化问题

## 数学处理（朗格朗日乘子、对偶问题和KKT条件）

## 松弛支持向量机

## 核函数

## 支持向量回归

## 



