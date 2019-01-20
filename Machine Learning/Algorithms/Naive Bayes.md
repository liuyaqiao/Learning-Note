# 贝叶斯分类器

这是一种基于贝叶斯定理和各个特征条件独立假设的分类方法。换句话说，他是在所有概率都已知的情况的理想情况下，去找到基于这些概率和误判损失来寻找最优的分类方法；

## 贝叶斯误差

贝叶斯误差是指在现有特征集上，任意可以基于特征输入进行随机输出的分类器所能达到最小误差。也可以叫做最小误差。

## 思路

根据历史经验，我们需要基于后验概率会得到更加准确的分类结果。我们还是依照机器学习传统思路，根据一个误差函数去推导分类器模型的过程去分析：

这是在样本x上的条件损失：

<a href="https://www.codecogs.com/eqnedit.php?latex=R(c_i|x)&space;=&space;\sum_{j&space;=&space;1}&space;^&space;{N}\lambda_{ij}P(c_j|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(c_i|x)&space;=&space;\sum_{j&space;=&space;1}&space;^&space;{N}\lambda_{ij}P(c_j|x)" title="R(c_i|x) = \sum_{j = 1} ^ {N}\lambda_{ij}P(c_j|x)" /></a>

我们的目的是去寻找一个分类准则，可以最小化这个条件损失，这时的分类起就叫贝叶斯分类器。这里的lambda选用0-1误差函数。

这里1 - R体现了通过机器学习产生模型的精度的上限。？

如果这里的lambda选用0-1误差函数，那么贝叶斯分类器即为：

<a href="https://www.codecogs.com/eqnedit.php?latex=h(x)&space;=&space;argmaxP(c|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(x)&space;=&space;argmaxP(c|x)" title="h(x) = argmaxP(c|x)" /></a>

即，对于每一个样本，都需要去选择能使后验概率最大的类别标记。

我们的后验概率可以根据bayes公式去计算

<a href="https://www.codecogs.com/eqnedit.php?latex=P(c|x)&space;=&space;\frac{P(x,c)}{P(x)}&space;=&space;\frac{P(c)P(x|c)}{P(x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(c|x)&space;=&space;\frac{P(x,c)}{P(x)}&space;=&space;\frac{P(c)P(x|c)}{P(x)}" title="P(c|x) = \frac{P(x,c)}{P(x)} = \frac{P(c)P(x|c)}{P(x)}" /></a>

分母的P(x)与类别标记无关，可以省去。可以发现，这里后验概率只与类先验概率、类条件概率有关。因此，问题就转化为了由样本分布去估计先验和类条件概率的问题。

如果满足属性条件独立性假设，则可以得出朴素贝叶斯分类器。我们说，对已知类别，每个属性都独立地对分类结果发生影响。

## 后验最大化预测

我们的贝叶斯公式可以重写为：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(c|x)&space;=&space;\frac{P(c)P(x|c)}{P(x)}&space;=&space;\frac{P(c)}{P(x)}&space;\prod&space;_{i&space;=&space;1}^{d}P(x_i|c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(c|x)&space;=&space;\frac{P(c)P(x|c)}{P(x)}&space;=&space;\frac{P(c)}{P(x)}&space;\prod&space;_{i&space;=&space;1}^{d}P(x_i|c)" title="P(c|x) = \frac{P(c)P(x|c)}{P(x)} = \frac{P(c)}{P(x)} \prod _{i = 1}^{d}P(x_i|c)" /></a>

根据这里，我们的bayes分类器可以写成：

<a href="https://www.codecogs.com/eqnedit.php?latex=h_{nb}&space;=&space;argmaxP(c)\prod&space;_{i&space;=&space;1}&space;^{d}P(x_i|c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{nb}&space;=&space;argmaxP(c)\prod&space;_{i&space;=&space;1}&space;^{d}P(x_i|c)" title="h_{nb} = argmaxP(c)\prod _{i = 1} ^{d}P(x_i|c)" /></a>

我们可以说，朴素贝叶斯分类器的训练过程就是基于训练集D来估计类先验概率P(c)，并为每一个属性估计条件概率P(x|c)。之后，我们通过计算出的后验概率的大小去决定类别。

在根据公式计算先验和类条件概率时，如果出现了次数为0等特殊情况，我们利用拉普拉斯平滑来处理，在分子分母上同时添加一个小的变量\lambda.

## 参数估计

对于朴素贝叶斯估计，我们需要去得到**类先验概率**和**类条件概率**，这两个值我们通常需要去进行估计。我们常用的方法有极大似然估计和贝叶斯估计；

极大似然估计与贝叶斯估计是统计中两种对模型的参数确定的方法，两种参数估计方法使用不同的思想。前者来自于频率派，认为参数是固定的，我们要做的事情就是根据已经掌握的数据来估计这个参数；而后者属于贝叶斯派，认为参数也是服从某种概率分布的，已有的数据只是在这种参数的分布下产生的。所以，直观理解上，极大似然估计就是假设一个参数θ，然后根据数据来求出这个θ. 而贝叶斯估计的难点在于p(θ) 需要人为设定，之后再考虑结合MAP （maximum a posterior）方法来求一个具体的θ. 

## 极大似然估计

我们认为参数是一个固定的值，我们通过极大似然估计去得到这些参数的估计值。
[参考](https://blog.csdn.net/zengxiantao1994/article/details/72787849)

利用已知的样本结果，反推最有可能（最大概率）导致这样结果的参数值。

  求最大似然估计量的一般步骤：
        
（1）写出似然函数；

（2）对似然函数取对数，并整理；

（3）求导数；

（4）解似然方程。

  最大似然估计的特点：

（1）比其他估计方法更加简单；

（2）收敛性：无偏或者渐近无偏，当样本数目增加时，收敛性质会更好；

（3）如果假设的类条件概率模型正确，则通常能获得较好的结果。但如果假设模型出现偏差，将导致非常差的估计结果。

## 贝叶斯估计

我们认为参数也是一个随机变量，它也服从着一定的分布。我们可以假设参数满足一个先验分布，来计算基于当前数据集时，参数的后验分布。我们要根据贝叶斯公式计算出后验概率，之后去比较后验概率的大小去选择参数。这时，分母是一个积分的形式。

最大似然估计和贝叶斯估计在样本数量趋于无穷的时候是一样的，我们选择的时候，一般要从计算复杂度、先验概率的可靠性等方面去考虑。比如，条件概率出现有不对称的状况，这时可能贝叶斯估计会更好，最大似然估计往往会忽略这些特点。

# 最优误差

这里指的是贝叶斯误差，它是应用贝叶斯分类规则的分类器的错误率。贝叶斯分类规则的一个性质是:在最小化分类错误率上是最优的。所以在分类问题中，贝叶斯错误率是一个分类器对某个类别所能达到的最**低的分类错误率**。

我们可以这样理解，我们要求的理想模型就是知道了预先的分布之后，再去预测。这也就是bayes决策的出发点，预先知道真实分布，去预测现实的数据分布。这样得到的误差就是贝叶斯误差，这在其他分类器中是最优误差也是不可避免的误差。

# 应用和条件

在
