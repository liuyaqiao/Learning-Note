# Ensemble Learning

## 方差和偏差

## 集成策略

- Boosting  

	它表示一族可将所学习器提升为强学习器的算法。它的个体学习器间存在强依赖关系、必须串行序列化生成。它的大体思路是：先从初始训练集中训练出一个分类器，再根据基学习器的表现对训练样本的分布进行一个调整，使得先前基学习器做错的训练样本在后续收受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器；如此反复执行，直到基学习器的数目达到了事先指定的T。它主要涉及两个观点：**加法模型**和**前向分布**一般形式为：

	<a href="https://www.codecogs.com/eqnedit.php?latex=F_M(x;P)&space;=&space;\sum_{m&space;=&space;1}^n\beta_mh(x;\alpha_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_M(x;P)&space;=&space;\sum{m&space;=&space;1}^n\beta_mh(x;\alpha_m)" title="F_M(x;P) = \sum_{m = 1}^n\beta_mh(x;\alpha_m)" /></a>

	h是一个个弱的分类器，a是弱分类器学习到的最优参数，beta是每一个弱分类器所占的比重，P是参数的组合。至于前向分布，就是说训练过程中，下一轮迭代产生的分类器是在上一轮的基础上训练得来的。也就是可以写成这样的形式：：
	
	<a href="https://www.codecogs.com/eqnedit.php?latex=F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;&plus;&space;\beta_mh(x;\alpha_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;&plus;&space;\beta_mh(x;\alpha_m)" title="F_m(x) = F_{m - 1}(x) + \beta_mh(x;\alpha_m)" /></a>

- Adaboost

	由于损失函数不同，boosting算法也有了很多不同类型。其中adaboost就是损失函数是指数损失的boosting算法。

	最小化指数损失函数为：
	
	<a href="https://www.codecogs.com/eqnedit.php?latex=l_{exp}(H|D)&space;=&space;E_{x\sim&space;D}[e^{-f(x)H(x)}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{exp}(H|D)&space;=&space;E_{x\sim&space;D}[e^{-f(x)H(x)}]" title="l_{exp}(H|D) = E_{x\sim D}[e^{-f(x)H(x)}]" /></a>
	
	可以证明，指数损失函数是原来0/1损失具有一致性的替代函数。把前向分布公式代入，可以得到：
	
	<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\sum_{i&space;=&space;1}^{N}&space;exp(-y_{i}F_m(x_i))&space;=&space;\sum_{i&space;=&space;1}^{N}&space;exp(-y_i(F_{m&space;-&space;1}(x)&space;&plus;&space;\alpha_mG_m(x)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\sum_{i&space;=&space;1}^{N}&space;exp(-y_{i}F_m(x_i))&space;=&space;\sum_{i&space;=&space;1}^{N}&space;exp(-y_i(F_{m&space;-&space;1}(x)&space;&plus;&space;\alpha_mG_m(x)))" title="L = \sum_{i = 1}^{N} exp(-y_{i}F_m(x_i)) = \sum_{i = 1}^{N} exp(-y_i(F_{m - 1}(x) + \alpha_mG_m(x)))" /></a>
	
	经过化简可以得到：
	
	<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\sum_{i&space;=&space;1}^{N}&space;w_{m,i}&space;exp(-y_i\alpha_mG_m(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\sum_{i&space;=&space;1}^{N}&space;w_{m,i}&space;exp(-y_i\alpha_mG_m(x_i))" title="L = \sum_{i = 1}^{N} w_{m,i} exp(-y_i\alpha_mG_m(x_i))" /></a>
	
	这里w就是每轮迭代的权重样本，依赖于前一轮迭代重分配。
	
	<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;=&space;exp(-y_i(F_{m&space;-&space;1}&space;(x)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;=&space;exp(-y_i(F_{m&space;-&space;1}&space;(x)))" title="w = exp(-y_i(F_{m - 1} (x)))" /></a>

	这时我们由当分类正确时，y = 1；分类错误时，y = -1; 把这个结果代入公式，可以得到：

	<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\sum_{i&space;=&space;1}^{N}w_{m,i}&space;exp(-\alpha_mG_m(x_i))_{True}&space;&plus;&space;\sum_{i&space;=&space;1}^{N}w_{m,i}&space;exp(\alpha_mG_m(x_i))_{False}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\sum_{i&space;=&space;1}^{N}w_{m,i}&space;exp(-\alpha_mG_m(x_i))_{True}&space;&plus;&space;\sum_{i&space;=&space;1}^{N}w_{m,i}&space;exp(\alpha_mG_m(x_i))_{False}" title="L = \sum_{i = 1}^{N}w_{m,i} exp(-\alpha_mG_m(x_i))_{True} + \sum_{i = 1}^{N}w_{m,i} exp(\alpha_mG_m(x_i))_{False}" /></a>


	它主要由三部分组成，weak base learning algorithm + optimal re-weighting factor + linear aggregation。
	可以很快得到第一项（base algorithm），因为只需要比0.5大，所以只需要logn的时间。

	**adaboost 为什么不容易overfit，它主要关注了降低偏差。**

- Stacking

	见学习结合法
	
- Bagging

	它通过**自助采样法**得到，具体的操作是：给定m个样本的数据集，我们先随机取出一个样本放入采样机中，再把该样本放回初始数据集，使得下次采样时该样本仍可能被选中。经过m次采样，我们可以得到m个样本的采样集，初始训练集中有的样本在采样集中多次出现，而有的没有出现。可以证明，由63.2的数据出现在采样集中。没有取到的数据可以作为验证集来对泛化性能进行包外估计。
	
	我们可以采样出T个含有m个训练样本的采样集，然后基于每一个采样集训练出一个基学习器。再将这些学习器集合，这就是bagging的思想。对于输出使用简单的投票法，对于回归任务使用平均法。它主要是降低方差，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果明显。

	个体学习器不存在很强的依赖关系、可同时生成的并行化方法。例子**随机森林**。
	
	**随机森林：**
	
	RF是以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。在RF中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含K的属性的子集，然后再从这个子集中选择一个最优属性用于划分。这里参数k的引入控制了随机性的程度，k = d则是传统决策树；若k = 1则是随机划分。推荐采用k = log2d

	RF简单，计算开销小，容易实现，它同时具有很好的效果。RF的多样性不仅仅存在于样本扰动，还来自于属性扰动，这使得泛化性能得到了进一步提升。
	
	学习器结合的好处：
	- 单个学习器可能因为误选导致泛化能力不好，但是多个学习器结合不会有这个问题（统计原因）
	- 单个学习器容易陷入到局部最小（计算原因）
	- 多个学习器可以使假设空间扩大，得到更好的近似（表示原因）
	
	在**学习器结合**的时候，我们常常会考虑三种方法：
	
		1.平均法
		
		一般会使用简单平均法，加权平均的性能不一定会比简单平均的效率高。如果个体学习器差异比较大，适合运用加权平均；而如果差不多的话，则简单的平均就可以起到比较好的效果。
		
		2.投票法
		
		这里一般考虑绝对多数投票、相对多数投票和加权投票。这里有一个拒绝的机制，所以可靠性更好
		
		3.学习法（stacking）
		
		它先从初始数据集中训练出初级学习器，然后生成一个新的数据集用于训练次级学习器。在这个数据集中，初始学习器的输出作为输入，而原数据样本的标记仍作为样例标记。初级学习器可以是由相同或不同的算法生成。
		
		这里要注意的是如果直接用初级学习器的训练集去产生次级训练集的话，则会过拟合风险比较大；所以，一般是通过交叉验证或者留一法，用训练初级学习器未使用的样本来产生次级学习器的训练样本。
		
## GBDT

- 算法

	**用一句话来概括：以决策树为基学习器的一种利用boosting方法集成的学习策略。**

	我们从boosting算法谈起，boosting算法的基本结构为：

	输入：训练数据集T = {(x1,y1), (x2,y2),...,(xn,yn)}; 损失函数L; 基函数集b：  
	输出：加法模型f(x)

	1.初始化  
	2.对于m = 1,2,...,M  
    	(a)最小化损失函数,得到更新参数（这里是两个最优化的嵌套，需要优化两个参数）  
    	(b)更新模型  
    	(c)得到加法模型  

	我们可以发现，当误差函数是指数误差的时候，就是adaboost的形式。我们这里如果只采用一般的误差，或者一些不好计算的误差（除去来logloss，平方误差等），在最小化Lossfun时出现了计算问题，这时我们就要考虑引入新的最优化方法：[最速下降法](https://zhuanlan.zhihu.com/p/32709034)，结论就是，函数的负梯度方向是当前下降速度最快的方向。据此，我们可以推导出一些基本的公式：

	根据boosting算法的定义，可得：

	<a href="https://www.codecogs.com/eqnedit.php?latex=F(x)&space;=&space;\sum\gamma&space;_ih_i(x)&space;&plus;&space;const" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(x)&space;=&space;\sum\gamma&space;_ih_i(x)&space;&plus;&space;const" title="F(x) = \sum\gamma _ih_i(x) + const" /></a>

	<a href="https://www.codecogs.com/eqnedit.php?latex=F_{m}&space;=&space;F_{m&space;-&space;1}&space;&plus;&space;h(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{m}&space;=&space;F_{m&space;-&space;1}&space;&plus;&space;h(x)" title="F_{m} = F_{m - 1} + h(x)" /></a>

	我们可以得到：

	<a href="https://www.codecogs.com/eqnedit.php?latex=F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;&plus;&space;argmin_{h_m}[\sum_{i&space;=&space;1}^nL(y_i,&space;F_{m&space;-&space;1}(x_i)&space;&plus;&space;h_m(x_i))]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;&plus;&space;argmin_{h_m}[\sum_{i&space;=&space;1}^nL(y_i,&space;F_{m&space;-&space;1}(x_i)&space;&plus;&space;h_m(x_i))]" title="F_m(x) = F_{m - 1}(x) + argmin_{h_m}[\sum_{i = 1}^nL(y_i, F_{m - 1}(x_i) + h_m(x_i))]" /></a>

	在任意的Loss函数时，我们在计算最小化的时候很困难。根据最速下降法，我们取负梯度的方向，可得：

	<a href="https://www.codecogs.com/eqnedit.php?latex=F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;-&space;\gamma&space;_m\sum_{i&space;=&space;1}^{n}\bigtriangledown&space;_{F_{m&space;-&space;1}}&space;L(y_i.&space;F_{m&space;-&space;1}(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_m(x)&space;=&space;F_{m&space;-&space;1}(x)&space;-&space;\gamma&space;_m\sum_{i&space;=&space;1}^{n}\bigtriangledown&space;_{F_{m&space;-&space;1}}&space;L(y_i.&space;F_{m&space;-&space;1}(x_i))" title="F_m(x) = F_{m - 1}(x) - \gamma _m\sum_{i = 1}^{n}\bigtriangledown _{F_{m - 1}} L(y_i. F_{m - 1}(x_i))" /></a>

	<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;_m&space;=&space;argmin_{\gamma}&space;\sum_{i&space;=&space;1}^nL(y_i,&space;F_{m&space;-&space;1}(x_i)&space;-&space;\gamma\bigtriangledown&space;_{F_{m&space;-&space;1}}L(y_i,&space;F_{m&space;-&space;1}(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;_m&space;=&space;argmin_{\gamma}&space;\sum_{i&space;=&space;1}^nL(y_i,&space;F_{m&space;-&space;1}(x_i)&space;-&space;\gamma\bigtriangledown&space;_{F_{m&space;-&space;1}}L(y_i,&space;F_{m&space;-&space;1}(x_i))" title="\gamma _m = argmin_{\gamma} \sum_{i = 1}^nL(y_i, F_{m - 1}(x_i) - \gamma\bigtriangledown _{F_{m - 1}}L(y_i, F_{m - 1}(x_i))" /></a>

	根据上式分别求出学习率和导数，就可以得到最终的F(x).

	算法说明如图：
	![gbdt](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/gdbt.png)
	
	谈一下**残差**：

	在boosting中，我们开始求解的公式为：

	<a href="https://www.codecogs.com/eqnedit.php?latex=F_{m&space;&plus;&space;1}(x)&space;=&space;F_m(x)&space;&plus;&space;h(x)&space;=&space;y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{m&space;&plus;&space;1}(x)&space;=&space;F_m(x)&space;&plus;&space;h(x)&space;=&space;y" title="F_{m + 1}(x) = F_m(x) + h(x) = y" /></a>

	<a href="https://www.codecogs.com/eqnedit.php?latex=h(x)&space;=&space;y&space;-&space;F_m(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(x)&space;=&space;y&space;-&space;F_m(x)" title="h(x) = y - F_m(x)" /></a>

	我们就把上面这个式子定义为残差，可以理解成：每一次的拟合都是在基于现在的残差去拟合的，这也是boosting慢慢变好的原理。在上面的算法中，我们用一个负梯度的形式代替了残差，这里可以根据函数在极小区间内的泰勒展开来理解。一个函数可以写成形如：f(x) = f(x - 1) + x区间 * 导数 的形式。

	到这里，我们导出了gradient boosting的形式，下面考虑把Decision Tree当作基学习器（CART）：

	gbdt在第m步的时候使用决策树去拟合残差，残差依旧是通过上述的残差公式计算。我们假定J表示子叶的个数，h表示决策树，R表示每一个子节点的区域，b表示在该叶节点中所属的值。则有：

	<a href="https://www.codecogs.com/eqnedit.php?latex=h_m(x)&space;=&space;\sum_{j&space;=&space;1}^{J_{m}}b_{jm}1_{R_{jm}}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_m(x)&space;=&space;\sum_{j&space;=&space;1}^{J_{m}}b_{jm}1_{R_{jm}}(x)" title="h_m(x) = \sum_{j = 1}^{J_{m}}b_{jm}1_{R_{jm}}(x)" /></a>

	这里1表示在对应的R中，取1；反之为0；

	其余对Fm和权值的求解和上式相同，依旧是去做一个前向加法的最优化问题，最后使用线性加权平均的方式去组合。如下图：

	![gdbt2](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/gbdt2.png)
	
	

- Regularization

	1.一个很重要的因素是轮次M，太高的M会导致overfit。（early stop）
	2.学习率，在F每一次更新的时候加一个学习率。我们会发现
当学习率小的时候，得到模型的泛化能力比较好，但是这需要比较大的迭代次数。（缩减）每次只学习一点点来减小单个特征对整体的影响
	3.叶结点的数量，对树的复杂程度进行约束（叶节点在全部节点中占据的比例）
	4.子采样（subsample）：采样比例（subsample）取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间。使用了子采样的GBDT有时也称作随机梯度提升树(Stochastic Gradient Boosting Tree, SGBT)。由于使用了子采样，程序可以通过采样分发到不同的任务去做boosting的迭代过程，最后形成新树，从而减少弱学习器难以并行学习的弱点。


- 优点

	1. 可以灵活处理各种类型的数据，包括连续值和离散值。
	2. 在相对少的调参时间情况下，预测的准备率也可以比较高。这个是相对SVM来说的。
	3. 使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数。

- 缺点

	GBDT的主要缺点有：由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行。



- 用GBDT去解决分类问题：

	GBDT的分类算法从思想上和GBDT的回归算法没有区别，但是由于样本输出不是连续的值，而是离散的类别，导致我们无法直接从输出类别去拟合类别输出的误差。为了解决这个问题，主要有两个方法，一个是用指数损失函数，此时GBDT退化为Adaboost算法。另一种方法是用类似于逻辑回归的对数似然损失函数的方法。依旧是计算残差，但是计算方式略有不同。

[参考文献](https://zhuanlan.zhihu.com/p/40096769)


