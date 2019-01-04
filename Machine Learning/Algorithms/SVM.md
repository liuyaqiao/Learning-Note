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
>以下是推导过程：  
![SVM](https://github.com/liuyaqiao/Learning-Note/blob/master/svm.png)  
如图所示，我们称距离超平面最近的这几个训练样本点称为支持向量。把直线取在支持向量的中间，我们可以认为所得的距离最大。  
假设超平面可以将分类样本分类正确，对于y = 1, 则有wx + b >= 1（这里根据距离公式可以得到，正确分类的距离r >= 1/||w||，之后加上label对样本的影响），若y = -1，则有wx + b <= -1。我们通过对距离公式中w的变换，可以得到：  

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." title="\left\{\begin{matrix} w^{T}x + b \geq +1, y_{i} = +1\\ w^{T}x + b \leq -1, y_{i} = -1 \end{matrix}\right." /></a>  
>这里是规定强行规定了1和-1的间隔值）。如果超平面总能将训练样本分开，则可以通过对w、b的线性变换和最优化函数的参数改变，使得距离变成1.  
通过对直线距离公式，构造出的两条直线的间隔是2/|w|,所以可以构造出的最优化问题为：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" title="\begin{matrix} max \quad\frac{2}{\left \| w \right \|}\\ s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}" /></a>  
同理，上述的最优化问题等价于：  

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." title="\left\{\begin{matrix} min\quad \frac{1}{2}||w||^{2}\\s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}\right." /></a>  
上式就是支持向量机的基本型。  
用通俗的话来讲，这是一个有约束条件的最优化问题。约束条件是样本空间，最优化的目标是我们取的分类标准。用一句话概括：svm是在当前样本空间中，寻找一个使得异类样本之间具有最大距离的超平面（分类器）。  
这里我们构成了一个凸优化的问题，因为所优化项本身是一个**凸二次规划问题**，可以有现成的包求解，但是我们具有更好的解决办法。（见下文）

## 数学处理（拉格朗日乘子、对偶问题和KKT条件）
首先说一下整体的推导思路：  
首先我们得到一个**有约束的最优化问题**，这时我们考虑已经的梯度下降（gradient descent）等方法去处理，但是由于条件的限制导致这个问题不是很好处理。所以我们要考虑讲问题转化成**没有约束条件的最优化问题**，这时我们采用了**拉格朗日乘子法**。  
我们本来可以通过求导的方法取求解这个问题，但是这时计算的过程过于复杂，所以我们考虑通过构造**拉格朗日对偶问题**来解决这个复杂计算的问题。  
为了保证原问题和对偶问题所求的结果一致，我们需要证明这个问题满足**强对偶关系**。  
经过证明之后，我们可以使用经过证明的对偶问题是原问题具有相同的解。  
而**KKT条件**，则是**不等式约束下的最优化问题有解**、**构造对偶问题**和证明**强对偶关系**都需要用到的条件，可以说是整个理论的一个纽带。  

总结来说，我们是通过构造一个满足强对偶关系的拉格朗日对偶问题来求解SVM。
  
1.  上文中我们提到[有约束的最优化问题](https://zhuanlan.zhihu.com/p/26514613)，见参考文献，我们这里属于在不等式约束下的最优化问题，它取得极值的必要条件应该是KKT条件。
>这里注明一下必要条件：
例如在有等式约束的最优化问题中，我们构造的拉格朗日量对约束量偏导为0是取得极值的必要条件指的是，取得函数极值我们一定有偏导数为0，但是偏导数为0不一定会取得极值，我们还需要根据偏导数对函数大小进行比较。而KKT条件则是不等式约束条件下的优化函数取得极值的必要条件。  
之后如果有时间会写一篇[KKT条件的推导]!!!!!!!!!!!!!!  
2.  拉格朗日乘子法  
我们根据拉格朗日乘子法可以得到如下的拉氏量:  
<a href="https://www.codecogs.com/eqnedit.php?latex=L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" title="L(w,b,a) = \frac{1}{2} \left \| w \right \|^{2} + \sum_{1}^{m}\alpha _{i}(1 - y_{i}(w^{T}x_{i} + b))" /></a>  
这里我们规定a均大于等于0.（？？？）
3.  对偶问题  
上述问题我们通过求导的方法仍然不容易求解，所以我们为了构造更加容易求解的最优化形式，来构造对偶问题。上式可以等价于:  
<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))" /></a>  
Proof:  
<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min_{b,w}max_{a_{n}\geq&space;0}(\frac{1}{2}||w||^{2}&space;&plus;&space;a_{n}&space;(1&space;-&space;y_{n}(w^{T}&space;x_{i}&space;&plus;&space;b_{i})))&space;\\&space;=&space;min_{b,&space;w}(\infty&space;\text{&space;if&space;violate};&space;\frac{1}{2}&space;||w||^{2}&space;\text{&space;if&space;feasible})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min_{b,w}max_{a_{n}\geq&space;0}(\frac{1}{2}||w||^{2}&space;&plus;&space;a_{n}&space;(1&space;-&space;y_{n}(w^{T}&space;x_{i}&space;&plus;&space;b_{i})))&space;\\&space;=&space;min_{b,&space;w}(\infty&space;\text{&space;if&space;violate};&space;\frac{1}{2}&space;||w||^{2}&space;\text{&space;if&space;feasible})" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))\\ = min_{b,w}max_{a_{n}\geq 0}(\frac{1}{2}||w||^{2} + a_{n} (1 - y_{n}(w^{T} x_{i} + b_{i}))) \\ = min_{b, w}(\infty \text{ if violate}; \frac{1}{2} ||w||^{2} \text{ if feasible})" /></a>  
如果有任何违反规则，第二项大于0，则最大值会趋紧于正无穷。  
如果所有的点都符合的话，第二项小于0，最大值则为0. 
这时我们已经把限制条件加入到了max中。即可得到，    
<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min(\frac{1}{2}||w||^{2}),&space;\text{meet&space;constrains}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min(\frac{1}{2}||w||^{2}),&space;\text{meet&space;constrains}" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))\\ = min(\frac{1}{2}||w||^{2}), \text{meet constrains}" /></a>  



4.  

我们试图找出这个凸优化问题的对偶问题：  
>对偶问题：  
一个优化问题可以从主问题和对偶问题两个角度来考虑。一般来说，对偶问题给出了最优值的下界，这个下界取决于拉格朗日函数引入的参数的取值。但是，基于对偶函数可以取到的最好的下界是什么？这是我们关注的问题，这里可以引入一个关于参数的max的最优化问题，这就是原问题的对偶问题，这一定是一个凸优化问题：  
对于原函数是凸函数的情况，在特定条件下，强对偶性成立（min最优值和max最优质相等 ），可以通过这个关系来求解主问题。

使用[拉格朗日乘子法](https://www.cnblogs.com/sddai/p/5728195.html)对x求偏导来得到对偶问题，可以得到：  
这里规定引入的参数均满足大于等于0:
<a href="https://www.codecogs.com/eqnedit.php?latex=L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" title="L(w,b,a) = \frac{1}{2} \left \| w \right \|^{2} + \sum_{1}^{m}\alpha _{i}(1 - y_{i}(w^{T}x_{i} + b))" /></a>

我们去试图找出这个问题的

[约束问题](https://zhuanlan.zhihu.com/p/26514613)

## 松弛支持向量机

## 核函数

## 支持向量回归

## 



