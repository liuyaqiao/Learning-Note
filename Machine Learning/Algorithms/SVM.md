# Support Vector Machine
>简介：
>定义在特征空间上的间隔最大的分类器。
## 出发点和思路
&ensp;&ensp;&ensp;&ensp;我们在空间中可以找到无数条线做分类，都可以达到分类的效果。但是分类器有好坏之分，我们的出发点是选择一个能起到最好的分类效果的分类器。从直观上看，我们应该寻找**在所有样本最中间的一条直线**做为最终的分类器，这样的分类器对噪音容忍度比较高，对未知数据的泛化能力比较强。  
## 基本型的推导
&ensp;&ensp;&ensp;&ensp;在样本空间，划分超平面可以通过如下的线性方程来描述：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=w^{T}x&space;&plus;&space;b&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{T}x&space;&plus;&space;b&space;=&space;0" title="w^{T}x + b = 0" /></a>,  

&ensp;&ensp;&ensp;&ensp;其中，w表示法向量决定了平面的方向，b是位移量也叫偏执量，决定了超平面和原点之间的距离。空间中任意一点到这个超平面的距离可以表示为：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;=&space;\frac{\left&space;|&space;w^{T}&space;x&space;&plus;&space;b&space;\right&space;|}{\left&space;\|&space;w&space;\right&space;\|}" title="r = \frac{\left | w^{T} x + b \right |}{\left \| w \right \|}" /></a>  

&ensp;&ensp;&ensp;&ensp;我们通过这个距离公式和约束条件就可以推导出SVM的基本型：  
>以下是推导过程：  
![SVM](https://github.com/liuyaqiao/Learning-Note/blob/master/svm.png)  
&ensp;&ensp;&ensp;&ensp;如图所示，如果把直线取在最近异类样本的中间，我们可以认为所得到的分类器对所有样本的容忍程度最好，用数学的描述就是经过最近异类样本点、并且与超平面平行的两条直线之间的距离最大，这就是我们最后要优化的函数。  
&ensp;&ensp;&ensp;&ensp;至于限制条件，我们不能只关注于分类器可以正确的将异类样本分开的特点，这仅仅线性分类器的任务。而SVM是更紧致的分类器，它需要满足对于所有分类标签为1和-1样本点，它们到超平面的距离都大于等于d(支持向量上的样本点到超平面的距离)。所以也有人说，SVM某种意义上可以看作是加了正则化的线性分类器（下文中会谈到）。这里的距离d = b/||w||，根据点到直线的距离公式并且考虑label对样本点的影响，我们可以得到：  
&ensp;&ensp;&ensp;&ensp;对于y = 1, 则有wx + b >= 1，  
&ensp;&ensp;&ensp;&ensp;对于y = -1，则有wx + b <= -1。则可以得到：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;w^{T}x&space;&plus;&space;b&space;\geq&space;&plus;1,&space;y_{i}&space;=&space;&plus;1\\&space;w^{T}x&space;&plus;&space;b&space;\leq&space;-1,&space;y_{i}&space;=&space;-1&space;\end{matrix}\right." title="\left\{\begin{matrix} w^{T}x + b \geq +1, y_{i} = +1\\ w^{T}x + b \leq -1, y_{i} = -1 \end{matrix}\right." /></a>   

&ensp;&ensp;&ensp;&ensp;通过对直线距离公式，构造出的两条直线的间隔是2/||w||,所以可以构造出的最优化问题为：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;max&space;\quad\frac{2}{\left&space;\|&space;w&space;\right&space;\|}\\&space;s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}" title="\begin{matrix} max \quad\frac{2}{\left \| w \right \|}\\ s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}" /></a>  

&ensp;&ensp;&ensp;&ensp;同理，上述的最优化问题等价于：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;min\quad&space;\frac{1}{2}||w||^{2}\\s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1,&space;\quad&space;i&space;=&space;1,2,3&space;\end{matrix}\right." title="\left\{\begin{matrix} min\quad \frac{1}{2}||w||^{2}\\s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1, \quad i = 1,2,3 \end{matrix}\right." /></a>  

&ensp;&ensp;&ensp;&ensp;上式就是支持向量机的基本型。  
&ensp;&ensp;&ensp;&ensp;用通俗的话来讲，这是一个有约束条件的最优化问题。约束条件是样本空间，最优化的目标是我们取的分类标准。用一句话概括：svm是在当前样本空间中，寻找一个使得异类样本之间具有最大距离的超平面（分类器）。  
&ensp;&ensp;&ensp;&ensp;这里我们构成了一个凸优化的问题，因为所优化项本身是一个**凸二次规划问题**，可以有现成的包求解，但是我们具有更好的解决办法。（见下文）

## 数学处理（拉格朗日乘子、对偶问题和KKT条件）
&ensp;&ensp;&ensp;&ensp;首先说一下整体的一种推导思路：  
&ensp;&ensp;&ensp;&ensp;首先我们得到一个**有约束的最优化问题**，这时我们考虑已经的梯度下降（gradient descent）等方法去处理，但是由于条件的限制导致这个问题不是很好处理。所以我们要考虑讲问题转化成**没有约束条件的最优化问题**，这时我们采用了**拉格朗日乘子法**。  
&ensp;&ensp;&ensp;&ensp;我们本来可以通过求导的方法取求解这个问题，但是这时计算的过程过于复杂，所以我们考虑通过构造**拉格朗日对偶问题**来解决这个复杂计算的问题。为了保证原问题和对偶问题所求的结果一致，我们需要证明这个问题满足**强对偶关系**。经过证明之后，我们可以把对偶问题的解当成是原问题的解。  
&ensp;&ensp;&ensp;&ensp;而**KKT条件**，则是**不等式约束下的最优化问题有解**、**构造对偶问题**和证明**强对偶关系**都需要用到的条件，可以说是整个理论的一个纽带。  

&ensp;&ensp;&ensp;&ensp;总结来说，我们是通过构造一个满足强对偶关系的拉格朗日对偶问题来求解SVM。下面分别介绍一下：**有约束的最优化问题、拉格朗日乘子法、拉格朗日对偶问题、强对偶问题和KKT**条件五个方面来介绍：  
  
1.有约束的最优化问题  
&ensp;&ensp;&ensp;&ensp;上文中我们提到[有约束的最优化问题](https://zhuanlan.zhihu.com/p/26514613)，见参考文献，我们这里属于在不等式约束下的最优化问题，它取得极值的必要条件应该是KKT条件。
>这里注明一下必要条件：
例如在有等式约束的最优化问题中，我们构造的拉格朗日量对约束量偏导为0是取得极值的必要条件指的是，取得函数极值我们一定有偏导数为0，但是偏导数为0不一定会取得极值，我们还需要根据偏导数对函数大小进行比较。而KKT条件则是不等式约束条件下的优化函数取得极值的必要条件。   

2.拉格朗日乘子法  
&ensp;&ensp;&ensp;&ensp;我们根据拉格朗日乘子法可以得到如下的拉氏量:  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" title="L(w,b,a) = \frac{1}{2} \left \| w \right \|^{2} + \sum_{1}^{m}\alpha _{i}(1 - y_{i}(w^{T}x_{i} + b))" /></a>  

&ensp;&ensp;&ensp;&ensp;这里我们规定a均大于等于0，这里可以理解为一个人为的限定，这个限定可以完美的符合最后我们推导出的一系列关系。

KKT条件为：

- 经过经过拉格朗日函数处理之后的新目标函数L(w,b,α)对x求导为零：

- <a href="https://www.codecogs.com/eqnedit.php?latex=h_j(x)=0；" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_j(x)=0；" title="h_j(x)=0；" /></a>
- <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha*g_k(k)=0&space;；" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha*g_k(k)=0&space;；" title="\alpha*g_k(k)=0 ；" /></a>


（后文中会详细讲述）    

对于我们的优化问题，条件二是满足的（相当于没有这一项）。也可以证明另外两个条件也满足，通过强对偶关系满足的KKT条件来证明。[参考](https://link.zhihu.com/?target=http%3A//blog.csdn.net/xianlingmao/article/details/7919597)

3.对偶问题  
>**对偶问题**：  
Dual problem 跟primal problem 可以看成本来是两个问题，因为优化的顺序不同而会得出两个不一定相关的值（但是minmaxf(x,y) >= maxminf(x,y)还是成立的，直观理解的话高中经常用的二次函数就可以了）。两者的差值就是duality gap，描述了我用另一种方式刻画问题的时候所造成的误差，强对偶的情况下最优值没有差别。在最优点处将会满足KKT 条件，但是KKT条件本身并不需要问题满足强对偶。([转自知乎atom Native](https://www.zhihu.com/question/58584814/answer/159079694))  

&ensp;&ensp;&ensp;&ensp;根据拉格朗日乘子的理论，我们把一个有约束的最优化问题转化成了一个无约束的最优化问题，相当于是对拉氏量进行优化，即min L(w,b,a)，但是用求导的方法求解它仍然不容易（因为这里参数数量多，并且出现了耦合，计算起来不方便），所以我们来构造更加简单的对偶的形式来求解（分离参数，并且用简单的办法求出一些参数）。该式可以等价于: 

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))" /></a>  

Proof:  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min_{b,w}max_{a_{n}\geq&space;0}(\frac{1}{2}||w||^{2}&space;&plus;&space;a_{n}&space;(1&space;-&space;y_{n}(w^{T}&space;x_{i}&space;&plus;&space;b_{i})))&space;\\&space;=&space;min_{b,&space;w}(\infty&space;\text{&space;if&space;violate};&space;\frac{1}{2}&space;||w||^{2}&space;\text{&space;if&space;feasible})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min_{b,w}max_{a_{n}\geq&space;0}(\frac{1}{2}||w||^{2}&space;&plus;&space;a_{n}&space;(1&space;-&space;y_{n}(w^{T}&space;x_{i}&space;&plus;&space;b_{i})))&space;\\&space;=&space;min_{b,&space;w}(\infty&space;\text{&space;if&space;violate};&space;\frac{1}{2}&space;||w||^{2}&space;\text{&space;if&space;feasible})" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))\\ = min_{b,w}max_{a_{n}\geq 0}(\frac{1}{2}||w||^{2} + a_{n} (1 - y_{n}(w^{T} x_{i} + b_{i}))) \\ = min_{b, w}(\infty \text{ if violate}; \frac{1}{2} ||w||^{2} \text{ if feasible})" /></a>  

&ensp;&ensp;&ensp;&ensp;如果有任何违反规则，第二项大于0，则最大值会趋近于正无穷。  
&ensp;&ensp;&ensp;&ensp;如果所有的点都符合的话，第二项小于0，最大值则为0.   
&ensp;&ensp;&ensp;&ensp;此时我们已经把限制条件加入到了max中，可以保证在符合约束条件的情况下，该构造的函数和原函数的优化结果相同。即可得到，    

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min(\frac{1}{2}||w||^{2}),&space;\text{meet&space;constrains}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{b,w}max_{a_{n}&space;\geqslant&space;0}(L(b,w,a))\\&space;=&space;min(\frac{1}{2}||w||^{2}),&space;\text{meet&space;constrains}" title="min_{b,w}max_{a_{n} \geqslant 0}(L(b,w,a))\\ = min(\frac{1}{2}||w||^{2}), \text{meet constrains}" /></a>  

&ensp;&ensp;&ensp;&ensp;我们交换max和min的顺序，得到这个问题的对偶问题为：  

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;max_{a}min_{b,w}\text{&space;}L(b,w,a)\\&space;s.t.\quad&space;a_{i}&space;\geq&space;0&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;max_{a}min_{b,w}\text{&space;}L(b,w,a)\\&space;s.t.\quad&space;a_{i}&space;\geq&space;0&space;\end{matrix}" title="\begin{matrix} max_{a}min_{b,w}\text{ }L(b,w,a)\\ s.t.\quad a_{i} \geq 0 \end{matrix}" /></a>  

&ensp;&ensp;&ensp;&ensp;根据对偶函数的性质（见周志华西瓜书附录405页），对偶函数给出了主问题最优值的下界，即有：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=maxminL(w,b,a)&space;\leq&space;minmaxL(w,b,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?maxminL(w,b,a)&space;\leq&space;minmaxL(w,b,a)" title="maxminL(w,b,a) \leq minmaxL(w,b,a)" /></a>  

&ensp;&ensp;&ensp;&ensp;我们只有证明这个函数满足**强对偶关系**之后，才可以将两个最优值之间画上等号。

4.强对偶问题的证明：
>强对偶问题：  
对于一个二次规划问题，强对偶关系存在有三个条件：  
    1.  凸函数（convex primal）  
    2.  有解（feasible primal）  
    3.  线性条件（linear constraints）    

&ensp;&ensp;&ensp;&ensp;这里也叫做slater条件，我们构造出的拉格朗日对偶问题满足了slater条件，即它是一个强对偶问题（具体的数学表达自行wiki）。可以证明这里的拉格朗日函数满足强对偶的条件，所以这里可以来解它的对偶问题。（满足强对偶问题的函数在取到最优值的时候均满足KKT条件，即KKT条件是强对偶问题的必要条件，这里林轩田老师的机器学习技法课程中给了更为详细的推导 ，本文就不详细展开，我们直接使用这个结论）

5.KKT条件：  
&ensp;&ensp;&ensp;&ensp;首先用一幅图来说明规范性条件、KKT条件和强对偶之间的关系：  
&ensp;&ensp;&ensp;&ensp;![KKT](https://github.com/liuyaqiao/Learning-Note/blob/master/RC_KKT_DUAL.png)  
&ensp;&ensp;&ensp;&ensp;(转自知乎用户徐林杰[link](https://zhuanlan.zhihu.com/p/36621652))  
&ensp;&ensp;&ensp;&ensp;大体来说，KKT条件是一个不等式约束条件取得极值的必要条件，但是KKT条件并不一定所有情况都满足。要满足KKT条件需要有一个规范性条件（Regularity conditions），为的是要求约束条件的质量不能太差。强对偶性质非常好，但是要求也很苛刻，比 KKT 条件要苛刻。如果问题满足强对偶一定也满足 KKT 条件，反之不一定。  
&ensp;&ensp;&ensp;&ensp;接着，我们来具体看一下KKT条件到底是什么：  
&ensp;&ensp;&ensp;&ensp;对于具有等式和不等式约束的一般优化问题： 

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;minf(x)\\&space;s.t.g_{j}(x)&space;\leq&space;0(j&space;=&space;1,2\cdot&space;\cdot&space;\\&space;h_{k}(x)&space;=&space;0(k&space;=&space;1,2,\cdot\cdot,l)&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;minf(x)\\&space;s.t.g_{j}(x)&space;\leq&space;0(j&space;=&space;1,2\cdot&space;\cdot&space;\\&space;h_{k}(x)&space;=&space;0(k&space;=&space;1,2,\cdot\cdot,l)&space;\end{matrix}" title="\begin{matrix} minf(x)\\ s.t.g_{j}(x) \leq 0(j = 1,2\cdot \cdot \\ h_{k}(x) = 0(k = 1,2,\cdot\cdot,l) \end{matrix}" /></a>  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;&&&space;\frac{\partial&space;f}{\partial&space;x_{i}}&space;&plus;&space;\sum_{j&space;=&space;1}^{m}\mu_{j}\frac{\partial&space;g_{i}}{\partial&space;x_{i}}&space;&plus;&space;\sum_{k&space;=&space;1}^{l}\lambda&space;_{k}\frac{\partial&space;h_{k}}{\partial&space;x_{i}}&space;=&space;0,&space;(i&space;=&space;1,2,...,n)&space;\\&space;&&&space;h_{k}(x)&space;=&space;0,&space;(k&space;=&space;1,2,...,l)&space;\\&space;&&&space;\mu&space;_{j}g_{j}&space;=&space;0,&space;(j&space;=&space;1,2,...,&space;m)&space;\\&space;&&&space;\mu_{j}&space;\geq&space;0&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;&&&space;\frac{\partial&space;f}{\partial&space;x_{i}}&space;&plus;&space;\sum_{j&space;=&space;1}^{m}\mu_{j}\frac{\partial&space;g_{i}}{\partial&space;x_{i}}&space;&plus;&space;\sum_{k&space;=&space;1}^{l}\lambda&space;_{k}\frac{\partial&space;h_{k}}{\partial&space;x_{i}}&space;=&space;0,&space;(i&space;=&space;1,2,...,n)&space;\\&space;&&&space;h_{k}(x)&space;=&space;0,&space;(k&space;=&space;1,2,...,l)&space;\\&space;&&&space;\mu&space;_{j}g_{j}&space;=&space;0,&space;(j&space;=&space;1,2,...,&space;m)&space;\\&space;&&&space;\mu_{j}&space;\geq&space;0&space;\end{matrix}\right." title="\left\{\begin{matrix} && \frac{\partial f}{\partial x_{i}} + \sum_{j = 1}^{m}\mu_{j}\frac{\partial g_{i}}{\partial x_{i}} + \sum_{k = 1}^{l}\lambda _{k}\frac{\partial h_{k}}{\partial x_{i}} = 0, (i = 1,2,...,n) \\ && h_{k}(x) = 0, (k = 1,2,...,l) \\ && \mu _{j}g_{j} = 0, (j = 1,2,..., m) \\ && \mu_{j} \geq 0 \end{matrix}\right." /></a>

&ensp;&ensp;&ensp;&ensp;接着，我们根据KKT条件是取得最优值的必要条件取解决对偶问题。

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w,b,a)&space;=&space;\frac{1}{2}&space;\left&space;\|&space;w&space;\right&space;\|^{2}&space;&plus;&space;\sum_{1}^{m}\alpha&space;_{i}(1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" title="L(w,b,a) = \frac{1}{2} \left \| w \right \|^{2} + \sum_{1}^{m}\alpha _{i}(1 - y_{i}(w^{T}x_{i} + b))" /></a>

&ensp;&ensp;&ensp;&ensp;对这个L氏量进行KKT条件的代入，分别对w,b求偏导数为0，可得： 

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L}{\partial&space;w}&space;=&space;0\rightarrow&space;w&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha&space;_{i}y_{i}x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;w}&space;=&space;0\rightarrow&space;w&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha&space;_{i}y_{i}x_{i}" title="\frac{\partial L}{\partial w} = 0\rightarrow w = \sum_{i = 1}^{n}\alpha _{i}y_{i}x_{i}" /></a>  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L}{\partial&space;b}&space;=&space;0\rightarrow&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}y_{i}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;b}&space;=&space;0\rightarrow&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}y_{i}&space;=&space;0" title="\frac{\partial L}{\partial b} = 0\rightarrow \sum_{i = 1}^{n}\alpha_{i}y_{i} = 0" /></a>  

&ensp;&ensp;&ensp;&ensp;再把这个结果带回到L中，可以得到：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=L(w,b,\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j&space;=&space;1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w,b,\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j&space;=&space;1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}" title="L(w,b,\alpha) = \sum_{i = 1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i,j = 1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}" /></a>  

&ensp;&ensp;&ensp;&ensp;到这里，内侧针对w,b的最小值已经求解完毕，现在要求解的是外侧的最大值，所以原来的优化问题可以转化为：  

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;L(w,b,\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j&space;=&space;1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}&space;\\&space;s.t.&space;\quad&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}y_{i}&space;=&space;1&space;\\&space;\alpha_{i}&space;\geq&space;0,&space;i&space;=&space;1,2,3...&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;L(w,b,\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i,j&space;=&space;1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}&space;\\&space;s.t.&space;\quad&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}y_{i}&space;=&space;1&space;\\&space;\alpha_{i}&space;\geq&space;0,&space;i&space;=&space;1,2,3...&space;\end{matrix}" title="\begin{matrix} L(w,b,\alpha) = \sum_{i = 1}^{n}\alpha_{i} - \frac{1}{2}\sum_{i,j = 1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} \\ s.t. \quad \sum_{i = 1}^{m} \alpha_{i}y_{i} = 1 \\ \alpha_{i} \geq 0, i = 1,2,3... \end{matrix}" /></a>

&ensp;&ensp;&ensp;&ensp;这就得出了原最优化问题的**对偶问题**，而且这里只有a一个变量，我们可以通过适当的最优化方法求出a之后（二次规划）。根据KKT条件，可以求出w和b，即可得到模型。

&ensp;&ensp;&ensp;&ensp;这里现在多采用比较流行的SMO方法取解决这个二次规划问题。

&ensp;&ensp;&ensp;&ensp;上面介绍了第一种思路，我们由slater条件出发，得到了一系列结果。还有一种被大家所接受的思路是：我们先提出规范性约束条件，再做一些更紧致的要求，使它进化为KKT条件。之后我们分析这个被优化函数的凹凸性和是否可微，可以判断出满足强对偶关系。针对这种思路我们这边不详细展开，大家可以去查阅相关资料和书籍。市面上大多教科书都是以这种思路来进行推导。

&ensp;&ensp;&ensp;&ensp;我们可以用一幅图来表达这些理论之间的关系：
![RL](https://github.com/liuyaqiao/Learning-Note/blob/master/Relationship.png)  


## 松弛支持向量机
&ensp;&ensp;&ensp;&ensp;之前我们讨论的情况都是假定在样本空间或特征空间是线性可分的，即存在一个超平面将不同类的样本完全划分开。但是在现实情况中，往往很难确定合适的核函数使得训练样本在特征空间线性可分；退一步说，即使找到这个线性可分的超平面，也很难确定这个貌似线性可分的结果是不是由过拟合所造成的。
&ensp;&ensp;&ensp;&ensp;缓解该问题的一个方案就是，允许SVM在一些样本上出错，为此我们引入了**软间隔**的概念。即，在两条直线之间可以允许出现一些分类错误的样本。与之前的**硬间隔**相对。它表示，某些样本不满足约束条件：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1" title="y_{i}(w^{T}x_{i} + b) \geq 1" /></a>


&ensp;&ensp;&ensp;&ensp;这时候我们的优化目标可以写成

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{w,b}&space;\frac{1}{2}&space;||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}l_{0/1}&space;(y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;-&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{w,b}&space;\frac{1}{2}&space;||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}l_{0/1}&space;(y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;-&space;1)" title="min_{w,b} \frac{1}{2} ||w||^{2} + C\sum_{i = 1}^{m}l_{0/1} (y_{i}(w^{T}x_{i} + b) - 1)" /></a>

&ensp;&ensp;&ensp;&ensp;这里C是一个常数，而l是一个0-1损失函数。

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=l_{0/1}&space;=&space;\left\{\begin{matrix}&space;1,&space;if&space;z<0&space;\\&space;0,&space;otherwise&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{0/1}&space;=&space;\left\{\begin{matrix}&space;1,&space;if&space;z<0&space;\\&space;0,&space;otherwise&space;\end{matrix}\right." title="l_{0/1} = \left\{\begin{matrix} 1, if z<0 \\ 0, otherwise \end{matrix}\right." /></a>

&ensp;&ensp;&ensp;&ensp;我们可以发现，根据C的取值不同，可以限制不满足条件样本的数目。这里，当C为有限值的时候，可以允许一些不满足的条件的样本。
&ensp;&ensp;&ensp;&ensp;我们发现，0-1误差函数非凸、非连续，数学性质不太好，代入优化函数中不好直接求解，所以我们想办法找出一些函数去替代l，这里我们称为替代损失（surrogate loss）。这些函数常常具有比较好的数学性质（连续，凸函数），并且都给出了0-1误差的上界。

- hinge loss 

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=l_{hinge}(z)&space;=&space;max(0,&space;1&space;-&space;z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{hinge}(z)&space;=&space;max(0,&space;1&space;-&space;z)" title="l_{hinge}(z) = max(0, 1 - z)" /></a>

- exponential loss

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=l_{exp}(z)&space;=&space;exp(-z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{exp}(z)&space;=&space;exp(-z)" title="l_{exp}(z) = exp(-z)" /></a>

- logistic loss

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=l_{log}(z)&space;=&space;log(1&space;&plus;&space;exp(-z))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{log}(z)&space;=&space;log(1&space;&plus;&space;exp(-z))" title="l_{log}(z) = log(1 + exp(-z))" /></a>

这里简要分析一下hinge的思路，如果出现了偏移量比较大的样本的时候，会对之前的0-1误差的损失函数带来比较大的影响。而hinge loss会尽量的减少这种情况带来的影响，它只是求了max(0, 1 - z)。针对远偏离的数据点，这里也只会取到max = 1，消除了这种outliner点对优化的影响。也可以证明，经过了hinge loss处理之后，优化的结果仍然只和支持向量有关，仍然保持了稀疏性。

&ensp;&ensp;&ensp;&ensp;我们在这里会采用hinge loss，则优化的公式变成：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{w,b}&space;\frac{1}{2}||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}max(0,&space;1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{w,b}&space;\frac{1}{2}||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}max(0,&space;1&space;-&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b))" title="min_{w,b} \frac{1}{2}||w||^{2} + C\sum_{i = 1}^{m}max(0, 1 - y_{i}(w^{T}x_{i} + b))" /></a>

&ensp;&ensp;&ensp;&ensp;如果引入松弛变量，可以讲上式重写为：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{w,b,\xi_{i}}&space;\frac{1}{2}||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}\xi&space;_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{w,b,\xi_{i}}&space;\frac{1}{2}||w||^{2}&space;&plus;&space;C\sum_{i&space;=&space;1}^{m}\xi&space;_{i}" title="min_{w,b,\xi_{i}} \frac{1}{2}||w||^{2} + C\sum_{i = 1}^{m}\xi _{i}" /></a>

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1&space;-&space;\xi_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.&space;\quad&space;y_{i}(w^{T}x_{i}&space;&plus;&space;b)&space;\geq&space;1&space;-&space;\xi_{i}" title="s.t. \quad y_{i}(w^{T}x_{i} + b) \geq 1 - \xi_{i}" /></a>

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\xi_{i}&space;\geq&space;0," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\xi_{i}&space;\geq&space;0," title="\xi_{i} \geq 0," /></a>

&ensp;&ensp;&ensp;&ensp;这就是软间隔支持向量机的基本形式。这里同样也属于二次规划问题，只是多了一个系数，这里就不多做展开。

&ensp;&ensp;&ensp;&ensp;我们还发现，如果替代损失采用不同的函数的话，会得到很多不同的结果，如果采用


- svm与lr
&ensp;&ensp;&ensp;&ensp;如果用对数几率损失函数去代替hinge损失函数则可以得到类似LR的误差函数形式。这实际上可以说明，SVM和LR的优化目标想尽，通常情况下，他们的性能也相当。

这两个损失函数的本质目的是一样的，都是为了增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。SVM的hinge的处理方法是，只考虑SV，去学习分类器。而LR是通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。SVM考虑了局部，而LR则考虑了全局。

[参考文献](https://blog.csdn.net/jfhdd/article/details/52319422)


- 误差分类

- svm与正则化




## 核函数
&ensp;&ensp;&ensp;&ensp;之前我们都假设所有样本是线性可分的，然而在现实问题中，我们不能确定样本可以在空间中线性可分。对于这样的问题，我们可以把样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分，我们的相应的模型可以写成：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;w^{T}\phi&space;(x)&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;w^{T}\phi&space;(x)&space;&plus;&space;b" title="f(x) = w^{T}\phi (x) + b" /></a>

&ensp;&ensp;&ensp;&ensp;类似的最优化问题可以写成：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=min_{w,b}&space;\frac{1}{2}||w||^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{w,b}&space;\frac{1}{2}||w||^{2}" title="min_{w,b} \frac{1}{2}||w||^{2}" /></a>

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.&space;\quad&space;y_{i}(w^{T}\phi(x_{i}&space;&plus;&space;b))\ge&space;1,&space;i&space;=&space;1,&space;2,&space;...m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.&space;\quad&space;y_{i}(w^{T}\phi(x_{i}&space;&plus;&space;b))\ge&space;1,&space;i&space;=&space;1,&space;2,&space;...m" title="s.t. \quad y_{i}(w^{T}\phi(x_{i} + b))\ge 1, i = 1, 2, ...m" /></a>

&ensp;&ensp;&ensp;&ensp;可以得出起对偶问题是：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=max_{a}&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i&space;=&space;1}^{m}\sum_{j&space;=&space;1}^{m}&space;\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_i)^T\phi(x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?max_{a}&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i&space;=&space;1}^{m}\sum_{j&space;=&space;1}^{m}&space;\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_i)^T\phi(x_j)" title="max_{a} \sum_{i = 1}^{m} \alpha_{i} - \frac{1}{2}\sum_{i = 1}^{m}\sum_{j = 1}^{m} \alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_i)^T\phi(x_j)" /></a>

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.&space;\sum_{i&space;=&space;1}^{m}\alpha_{i}y_{i}&space;=&space;0,&space;\\&space;\alpha_{i}&space;\ge&space;0,&space;i&space;=&space;1,2,...m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.&space;\sum_{i&space;=&space;1}^{m}\alpha_{i}y_{i}&space;=&space;0,&space;\\&space;\alpha_{i}&space;\ge&space;0,&space;i&space;=&space;1,2,...m" title="s.t. \sum_{i = 1}^{m}\alpha_{i}y_{i} = 0, \\ \alpha_{i} \ge 0, i = 1,2,...m" /></a>

&ensp;&ensp;&ensp;&ensp;凡是设计到\phi做内积的运算都很复杂，我们为了避开这个障碍，可以设想这样一个函数：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=\kappa&space;(x_i,x_j)&space;=&space;<\phi(x_i),\pha(x,j)>&space;=&space;\phi(x_i)^T\phi(x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\kappa&space;(x_i,x_j)&space;=&space;<\phi(x_i),\pha(x,j)>&space;=&space;\phi(x_i)^T\phi(x_j)" title="\kappa (x_i,x_j) = <\phi(x_i),\pha(x,j)> = \phi(x_i)^T\phi(x_j)" /></a>

&ensp;&ensp;&ensp;&ensp;我们通过这个函数来计算高维空间里面的内积，从而避开多维映射的计算。于是我们的对偶问题可以写成：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=max_{a}&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i&space;=&space;1}^{m}\sum_{j&space;=&space;1}^{m}&space;\alpha_{i}\alpha_{j}y_{i}y_{j}\kappa(x_i,x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?max_{a}&space;\sum_{i&space;=&space;1}^{m}&space;\alpha_{i}&space;-&space;\frac{1}{2}\sum_{i&space;=&space;1}^{m}\sum_{j&space;=&space;1}^{m}&space;\alpha_{i}\alpha_{j}y_{i}y_{j}\kappa(x_i,x_j)" title="max_{a} \sum_{i = 1}^{m} \alpha_{i} - \frac{1}{2}\sum_{i = 1}^{m}\sum_{j = 1}^{m} \alpha_{i}\alpha_{j}y_{i}y_{j}\kappa(x_i,x_j)" /></a>

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.&space;\sum_{i&space;=&space;1}^{m}\alpha_{i}y_{i}&space;=&space;0,&space;\\&space;\alpha_{i}&space;\ge&space;0,&space;i&space;=&space;1,2,...m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.&space;\sum_{i&space;=&space;1}^{m}\alpha_{i}y_{i}&space;=&space;0,&space;\\&space;\alpha_{i}&space;\ge&space;0,&space;i&space;=&space;1,2,...m" title="s.t. \sum_{i = 1}^{m}\alpha_{i}y_{i} = 0, \\ \alpha_{i} \ge 0, i = 1,2,...m" /></a>

&ensp;&ensp;&ensp;&ensp;求解后可以得到：

&ensp;&ensp;&ensp;&ensp;<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;w^{T}\phi(x)&space;&plus;&space;b&space;=&space;\sum_{i&space;=&space;1}&space;^&space;{m}\alpha_iy_i\kappa(x,x_i)&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;w^{T}\phi(x)&space;&plus;&space;b&space;=&space;\sum_{i&space;=&space;1}&space;^&space;{m}\alpha_iy_i\kappa(x,x_i)&space;&plus;&space;b" title="f(x) = w^{T}\phi(x) + b = \sum_{i = 1} ^ {m}\alpha_iy_i\kappa(x,x_i) + b" /></a>

&ensp;&ensp;&ensp;&ensp;这里的k(.,.)就称为和函数。有关核函数的性质和常用的核函数这里不过赘述，感兴趣可以查阅相关书籍和博客，这里比较简单就不做大段的展开。


## 支持向量回归

## 



