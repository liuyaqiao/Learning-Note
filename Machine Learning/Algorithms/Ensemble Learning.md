# Ensemble Learning

## 方差和偏差

## 集成策略

- Boosting
    它表示一族可将所学习器提升为强学习器的算法。它的个体学习器间存在强依赖关系、必须串行序列化生成。它的大体思路是：先从初始训练集中训练出一个分类器，再根据基学习器的表现对训练样本的分布进行一个调整，使得先前基学习器做错的训练样本在后续收受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器；如此反复执行，直到基学习器的数目达到了事先指定的T。它主要涉及两个观点：**加法模型**和**前向分布**一般形式为：

    <a href="https://www.codecogs.com/eqnedit.php?latex=F_M(x;P)&space;=&space;\sum_{m&space;=&space;1}^n\beta_mh(x;\alpha_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_M(x;P)&space;=&space;\sum_{m&space;=&space;1}^n\beta_mh(x;\alpha_m)" title="F_M(x;P) = \sum_{m = 1}^n\beta_mh(x;\alpha_m)" /></a>

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

adaboost 为什么不容易overfit，它主要关注了降低偏差。

- Stacking
    见学习结合法
    
- Bagging
    它通过自助采样法得到，具体的操作是：给定m个样本的数据集，我们先随机取出一个样本放入采样机中，再把该样本放回初始数据集，使得下次采样时该样本仍可能被选中。经过m次采样，我们可以得到m个样本的采样集，初始训练集中有的样本在采样集中多次出现，而有的没有出现。可以证明，由63.2的数据出现在采样集中。没有取到的数据可以作为验证集来对泛化性能进行包外估计。
    
    我们可以采样出T个含有m个训练样本的采样集，然后基于每一个采样集训练出一个基学习器。再将这些学习器集合，这就是bagging的思想。对于输出使用简单的投票法，对于回归任务使用平均法。它主要是降低方差，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果明显。

    个体学习器不存在很强的依赖关系、可同时生成的并行化方法。例子**随机森林**。
    
    随机森林：
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
        **这里要注意的是**如果直接用初级学习器的训练集去产生次级训练集的话，则会过拟合风险比较大；所以，一般是通过交叉验证或者留一法，用训练初级学习器未使用的样本来产生次级学习器的训练样本。
## GBDT

1. Boosting Tree

对于分类问题，我们采取cross entropy误差来解决，所以和adaboost的思路相同。之后考虑回归树，我们使用平方损失来解决。

<a href="https://www.codecogs.com/eqnedit.php?latex=L(y,&space;f(x))&space;=&space;(y&space;-&space;f(x))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(y,&space;f(x))&space;=&space;(y&space;-&space;f(x))^2" title="L(y, f(x)) = (y - f(x))^2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(y,&space;f_{m&space;-&space;1}(m)&space;&plus;&space;T(x,\theta_m))\\&space;=&space;[y&space;-&space;f_{m&space;-&space;1}(x)&space;-&space;T(x;&space;\theta_m)]^2&space;\\&space;=[r&space;-&space;T(x;\theta_m)]^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(y,&space;f_{m&space;-&space;1}(m)&space;&plus;&space;T(x,\theta_m))\\&space;=&space;[y&space;-&space;f_{m&space;-&space;1}(x)&space;-&space;T(x;&space;\theta_m)]^2&space;\\&space;=[r&space;-&space;T(x;\theta_m)]^2" title="L(y, f_{m - 1}(m) + T(x,\theta_m))\\ = [y - f_{m - 1}(x) - T(x; \theta_m)]^2 \\ =[r - T(x;\theta_m)]^2" /></a>

这里r我们称为残差。所以对回归问题的提升树算法来说，只需要拟合当前模型的残差。学习到的新的决策树要用残差代替之前的y,去拟合一个是当前损失函数最小的新的决策树。

2. Gradient Boosting
    当损失函数是平方损失和指数损失的时候，每一步优化都很简单。但对于一般损失而言，往往每一步优化都并不那么容易，所以有人提出用损失函数的负梯度来作为残差的一个近似值，来拟合一个回归树。即：

<a href="https://www.codecogs.com/eqnedit.php?latex=-[\frac{\partial&space;L(y,f(x_i))}{\partial&space;f(x_i)}]_{f(x))&space;=&space;f_{m-1}(x))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-[\frac{\partial&space;L(y,f(x_i))}{\partial&space;f(x_i)}]_{f(x))&space;=&space;f_{m-1}(x))}" title="-[\frac{\partial L(y,f(x_i))}{\partial f(x_i)}]_{f(x)) = f_{m-1}(x))}" /></a>

3. GBDT 和 Adaboost


GBDT 它的非线性变换比较多，表达能力强，而且不需要做复杂的特征工程和特征变换。
GBDT 的缺点也很明显，Boost 是一个串行过程，不好并行化，而且计算复杂度高，同时不太适合高维稀疏特征；
传统 GBDT 在优化时只用到一阶导数信息。
