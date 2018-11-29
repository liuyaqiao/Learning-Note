# Support Vector Machine
>简介：
>定义在特征空间上的间隔最大的分类器。
## 出发点和思路
我们在空间中可以找到无数条线做分类，都可以达到分类的效果。但是分类器有好坏之分，我们的出发点是选择一个能起到最好的分类效果的分类器。这就是我们的出发点。从直观上看，我们应该寻找在所有样本最中间的一条直线做为最终的分类器，这样的分类器对噪音容忍度比较高，对未知数据的泛化能力比较强。  
## 间隔公式
在样本空间，划分超平面可以通过如下的线性方程来描述：  
<a href="https://www.codecogs.com/eqnedit.php?latex=w^{T}x&space;&plus;&space;b&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{T}x&space;&plus;&space;b&space;=&space;0" title="w^{T}x + b = 0" /></a>,  
其中，w表示法向量决定了平面的方向，b是位移量也叫偏执量，决定了超平面和原地之间的距离。空间中任意一点在这个超平面的距离可以表示为：
<a href="https://www.codecogs.com/eqnedit.php?latex=r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" title="r = \frac{\left | w^{T} x + b \right |}{\left \| w \right \|}" /></a>  
>这里的公式可以通过一个线段到平面的投影来证明。  
![SVM](https://github.com/liuyaqiao/Learning-Note/blob/master/svm.png)  
如图所示，我们称距离超平面最近的这几个训练样本点称为支持向量。把直线取在支持向量的中间，我们可以认为所得的距离最大。  
假设超平面可以将分类样本分类正确，对于y = 1, 则有wx + b > 0，若y = -1，则有wx + b < 0。我们通过对距离公式中w的变换，可以得到：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." title="\left\{\begin{matrix} w^{T}x + b \geq +1, y_{i} = +1\\ w^{T}x + b \leq -1, y_{i} = -1 \end{matrix}\right." /></a>  
>这里是规定强行规定了1和-1的间隔值。如果超平面总能将训练样本分开，则可以通过对w、b的线性变换使得距离变成1.  
通过对直线距离公式，构造出的两条直线的间隔是2/|w|,所以可以构造出的最优化问题为：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" title="\begin{matrix} max \quad\frac{2}{\left \| w \right \|}\\ s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}" /></a>  
同理，上述的最优化问题等价于：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." title="\left\{\begin{matrix} min\quad \frac{1}{2}||w||^{2}\\s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}\right." /></a>  
上式就是支持向量机的基本型。  






-----------------
我们规定函数的几何间隔和函数间隔：
几何间隔：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;_{i}&space;=&space;y_{i}(wx_{i}&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;_{i}&space;=&space;y_{i}(wx_{i}&space;&plus;&space;b)" title="\gamma _{i} = y_{i}(wx_{i} + b)" /></a>  
函数间隔：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_{i}&space;=&space;y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|&space;}&space;x_{i}&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_{i}&space;=&space;y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|&space;}&space;x_{i}&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})" title="\gamma_{i} = y_{i}(\frac{w}{\left \| w \right \| } x_{i} + \frac{b}{\left \| w \right \|})" /></a>  
----------------
## 构造的凸优化问题

## 数学处理（朗格朗日乘子、对偶问题和KKT条件）

## 松弛支持向量机

## 核函数

## 支持向量回归

## 



