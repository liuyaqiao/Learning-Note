# Logistic Regression
## 推导逻辑
`逻辑回归是一种线性、以sigmoid作为判别函数的二元（多元）分类器`

前提：LR使用了线性组合的方式来逼近数据的分布，即可以使用wx的形式来表示数据的状态。

LR遵从的是的是伯努利分布，伯努利分布是一个离散的两点分布，结果只有是或否两种情况。  

有如下的随机变量概率分布：  

<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[x&space;=&space;1]&space;=&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[x&space;=&space;1]&space;=&space;p" title="Pr[x = 1] = p" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[x&space;=&space;0]&space;=&space;1&space;-&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[x&space;=&space;0]&space;=&space;1&space;-&space;p" title="Pr[x = 0] = 1 - p" /></a>

 我们考虑事件发生的对数几率：（事件发生与不发生的比值）  
 <a href="https://www.codecogs.com/eqnedit.php?latex=logit&space;=&space;log(\frac{p}{1-p})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logit&space;=&space;log(\frac{p}{1-p})" title="logit = log(\frac{p}{1-p})" /></a>  
我们可以采用线性组合来表示对数几率（依据上文）：  
<a href="https://www.codecogs.com/eqnedit.php?latex=log(\frac{p}{1-p})&space;=&space;wx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(\frac{p}{1-p})&space;=&space;wx" title="log(\frac{p}{1-p}) = wx" /></a>  
通过上面的表达式我们可以求出：  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 1|x) = \frac{e^{wx}}{e^{wx} + 1}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 0|x) = \frac{1}{e^{wx} + 1}" /></a>  
上面两个表达式的大小决定的测试例的归属类别，所以我们为了更方便的判断，我们选择了sigmoid函数。这里考虑到了sigmoid对称性好、值域在[0,1]，可以映射概率。sigmoid函数为：  
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=\frac{1}{1&space;&plus;&space;e^{-z}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=\frac{1}{1&space;&plus;&space;e^{-z}}" title="f(x) =\frac{1}{1 + e^{-z}}" /></a>
## 参数模型估计
我们使用`最大似然估计`：似然函数其实是参数关于w的函数，所表达的意思是当样本确定时，参数取某些值是的概率。所谓极大似然即是，参数最有可能出现的值。  
令  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[Y&space;=&space;1|x]&space;=&space;\pi&space;(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[Y&space;=&space;1|x]&space;=&space;\pi&space;(x)" title="Pr[Y = 1|x] = \pi (x)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[Y&space;=&space;0|x]&space;=&space;1&space;-&space;\pi&space;(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[Y&space;=&space;0|x]&space;=&space;1&space;-&space;\pi&space;(x)" title="Pr[Y = 0|x] = 1 - \pi (x)" /></a>  
可以得到对数似然函数：  
<a href="https://www.codecogs.com/eqnedit.php?latex=likelihood&space;=&space;\prod_{i}^{N}[\pi&space;(x_{i})]^{y_{i}}\cdot&space;[1&space;-&space;\pi&space;(x_{i})]^{1&space;-&space;y_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?likelihood&space;=&space;\prod_{i}^{N}[\pi&space;(x_{i})]^{y_{i}}\cdot&space;[1&space;-&space;\pi&space;(x_{i})]^{1&space;-&space;y_{i}}" title="likelihood = \prod_{i}^{N}[\pi (x_{i})]^{y_{i}}\cdot [1 - \pi (x_{i})]^{1 - y_{i}}" /></a>  
其对数似然函数为：  
<a href="https://www.codecogs.com/eqnedit.php?latex=L(w)&space;=&space;\sum_{1}^{N}[y_{i}log\pi&space;(x_{i})&space;&plus;&space;(1&space;-&space;y_{i})log(1&space;-&space;\pi&space;(x_{i}))]&space;=&space;\sum_{1}^{N}[y_{i}(wx_{i})&space;-&space;log(1&space;&plus;&space;exp(wx_{i})]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w)&space;=&space;\sum_{1}^{N}[y_{i}log\pi&space;(x_{i})&space;&plus;&space;(1&space;-&space;y_{i})log(1&space;-&space;\pi&space;(x_{i}))]&space;=&space;\sum_{1}^{N}[y_{i}(wx_{i})&space;-&space;log(1&space;&plus;&space;exp(wx_{i})]" title="L(w) = \sum_{1}^{N}[y_{i}log\pi (x_{i}) + (1 - y_{i})log(1 - \pi (x_{i}))] = \sum_{1}^{N}[y_{i}(wx_{i}) - log(1 + exp(wx_{i})]" /></a>

对于上式求最大值，我们可以利用`梯度下降`、`牛顿法`等凸优化算法（见后续的文档专门有介绍）去优化，求得最佳w的估计值，求出最优的w之后，我们就可以说，学习到的LR模型是：  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 1|x) = \frac{e^{wx}}{e^{wx} + 1}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 0|x) = \frac{1}{e^{wx} + 1}" /></a>    

## 代价函数（LogLoss）  

实际是，代价函数和最大似然估计本质上是等价的。

- 损失函数是一种用来衡量预测错误程度的函数，机器学习的基本策略就是使得经验风险最小化，也就是模型在训练上的损失最小，这是一种普适的思路。

- 而对于涉及到概率的问题，极大似然估计也是使得估计错误最小的一种思路，从数学推导上来看，LR的经验风险就是极大似然估计取个对数加个负号而已。

这是两种思维方式，但本质上是一样的，损失函数这一套逻辑是机器学习的普适逻辑，而极大似然估计这套思想是来解决概率相关问题。


<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta&space;)&space;=&space;-\frac{l(\theta&space;)}{m}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta&space;)&space;=&space;-\frac{l(\theta&space;)}{m}" title="J(\theta ) = -\frac{l(\theta )}{m}" /></a>  是似然函数对样本数商值得相反数  
我们为了得到最优得参数，需要最优化上述得代价函数。（这里以随机梯度下降法为例）  
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{j}&space;=&space;\theta_{j}&space;-&space;\alpha&space;\cdot&space;\frac{\delta&space;}{\delta&space;_{\theta_{j}}}J(\theta&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_{j}&space;=&space;\theta_{j}&space;-&space;\alpha&space;\cdot&space;\frac{\delta&space;}{\delta&space;_{\theta_{j}}}J(\theta&space;)" title="\theta_{j} = \theta_{j} - \alpha \cdot \frac{\delta }{\delta _{\theta_{j}}}J(\theta )" /></a>  
经过数学运算可以得到：  
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{j}&space;=&space;\theta_{j}&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i&space;=&space;1}^{m}(\pi(x_{i})&space;-&space;y_{i}))x_{j}^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_{j}&space;=&space;\theta_{j}&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i&space;=&space;1}^{m}(\pi(x_{i})&space;-&space;y_{i}))x_{j}^{i}" title="\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i = 1}^{m}(\pi(x_{i}) - y_{i}))x_{j}^{i}" /></a>  
末尾的x表示第i个pai(x)的值对第j个参数theta的导数，它就是对应位置某一个的x。[此处难以理解，后续将以代码展示]  

```
def gradAscent(dataMatIn, classLabels):
    #转换为numpy型
    dataMatrix = mat(dataMatIn) 
    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转制[[0],[1],[0].....] 
    # transpose() 行列转置函数
    # 将行向量转化为列向量   =>  矩阵的转置
    labelMat = mat(classLabels).transpose()
    # m->数据量，样本数 n->特征数
    m,n = shape(dataMatrix)
    alpha = 0.001 #步长
    maxCycles = 500 #迭代次数
    #初始化权值向量，每个维度均为1.0
    weights = ones((n,1))
    for k in range(maxCycles):
        #求当前sigmoid函数的值
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return array(weights)
```
这里可以将x看成一个矩阵，而参数是列向量，所内积的结果也是一个列向量。对相关参数求偏导的结果就是对应的xij元素。求和符号将所有的x合成了一个矩阵，所以写出了如上的代码。

## 正则化
如果变量（特征）过多的时候，会出现过拟合的情况。这种情况下训练出的方程总是能很好的拟合训练数据，也就是说，我们的代价函数可能非常接近于 0 或者就为 0。但是，这样做出的模型往往不够泛化，不能适用于大多数情况。为了防止这样的情况出现，我们会在目标函数中加入正则项，目的是使参数空间受到一定的限制。  
L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。  
<a href="https://www.codecogs.com/eqnedit.php?latex=J(w)&space;=&space;J(w)&space;&plus;&space;\lambda&space;\left&space;\|&space;w&space;\right&space;\|_{p}\left\{\begin{matrix}&space;L_{1}\rightarrow&space;\left&space;\|&space;w&space;\right&space;\|_{1}\\L2\rightarrow&space;\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|_{2}&space;^{2}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(w)&space;=&space;J(w)&space;&plus;&space;\lambda&space;\left&space;\|&space;w&space;\right&space;\|_{p}\left\{\begin{matrix}&space;L_{1}\rightarrow&space;\left&space;\|&space;w&space;\right&space;\|_{1}\\L2\rightarrow&space;\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|_{2}&space;^{2}&space;\end{matrix}\right." title="J(w) = J(w) + \lambda \left \| w \right \|_{p}\left\{\begin{matrix} L_{1}\rightarrow \left \| w \right \|_{1}\\L2\rightarrow \frac{1}{2}\left \| w \right \|_{2} ^{2} \end{matrix}\right." /></a>  
它们分别限制了参数的更新，

![](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/regularization.png)  
途中实线为参数w的等值线，虚线是L函数的图线。他们第一次相交的地方是最有解存在的取值。

给我们的直观感受是L2对参数的正则更加的平滑，即它限制了参数空间，但对参数的影响是平滑的，不像L1那样，直接使得某些参数的取值为0。  

总结起来就是：L1会引入`稀疏性`，而L2会充分利用更多的特征。  
>为什么L1会引入稀疏性呢？  
1.图形角度  
从上图中可以看到，L1的的两条线大部分会在矩形角的位置相交。这个位置会有w的分量为0。试想，在高维w分量的时候，就会出现多个w分量为0的情况。从而导致稀疏矩阵的出现。而L2正则化就没有这样的性质，所以不会有太多的稀疏性出现。  
2.先验角度      
正则化等价于对模型参数引入了先验分布，L1 Regularization 是认为模型参数服从拉普拉斯分布，L2 Regularization
则认为模型参数服从高斯分布（这里为了方便，使用线性回归的误差函数来说明）。其公式为:  
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{\sigma&space;\sqrt{2\pi&space;}}e^{-\frac{(x&space;-&space;\mu&space;)^{2}}{2\sigma^{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{\sigma&space;\sqrt{2\pi&space;}}e^{-\frac{(x&space;-&space;\mu&space;)^{2}}{2\sigma^{2}}}" title="f(x) = \frac{1}{\sigma \sqrt{2\pi }}e^{-\frac{(x - \mu )^{2}}{2\sigma^{2}}}" /></a>  
我们通过去对数似然函数的方法，可以分离出一个平方项。这就是L2的来源。而针对L1的数据，他们遵循拉普拉斯分布。
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{2\lambda&space;}e^{-\frac{\left&space;|&space;x&space;-&space;\mu&space;\right&space;|}{\lambda&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{2\lambda&space;}e^{-\frac{\left&space;|&space;x&space;-&space;\mu&space;\right&space;|}{\lambda&space;}}" title="f(x) = \frac{1}{2\lambda }e^{-\frac{\left | x - \mu \right |}{\lambda }}" /></a>  
![Laplace](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/laplace.jpg)  
拉普拉斯分布所得到的对数似然函数的附加项是线性，也就是L1的形式。  

>`注:` `数学推导`  
    规则化 = 加入参数的先验信息  
    我们认为w满足拉普拉斯分布或者高斯分布，不是单纯的天马行空的取w的值，而是有一个限制。  
    <a href="https://www.codecogs.com/eqnedit.php?latex=L(w)&space;=&space;p(y|X;w)p(w)=\prod_{i&space;=&space;1}^{m}p(y^{i}|x^{i};\theta)p(w)&space;\\=&space;\prod_{i&space;=&space;1}^{m}&space;\frac{1}{\sqrt{2\pi}\delta&space;}&space;exp(-\frac{(y^{i}&space;-&space;w^{T}x^{i})^{2}}{2\delta^{2}})&space;\frac{1}{\sqrt{2\pi}\alpha&space;}&space;exp(-&space;\frac{w^{T}w}{2\alpha&space;})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w)&space;=&space;p(y|X;w)p(w)=\prod_{i&space;=&space;1}^{m}p(y^{i}|x^{i};\theta)p(w)&space;\\=&space;\prod_{i&space;=&space;1}^{m}&space;\frac{1}{\sqrt{2\pi}\delta&space;}&space;exp(-\frac{(y^{i}&space;-&space;w^{T}x^{i})^{2}}{2\delta^{2}})&space;\frac{1}{\sqrt{2\pi}\alpha&space;}&space;exp(-&space;\frac{w^{T}w}{2\alpha&space;})" title="L(w) = p(y|X;w)p(w)=\prod_{i = 1}^{m}p(y^{i}|x^{i};\theta)p(w) \\= \prod_{i = 1}^{m} \frac{1}{\sqrt{2\pi}\delta } exp(-\frac{(y^{i} - w^{T}x^{i})^{2}}{2\delta^{2}}) \frac{1}{\sqrt{2\pi}\alpha } exp(- \frac{w^{T}w}{2\alpha })" /></a>  
    两边取对数之后可以得到：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=l(w)&space;=&space;logL(w)&space;\\=mlog\frac{1}{\sqrt{2\pi&space;}\delta&space;}&space;&plus;&space;n&space;log&space;\frac{1}{\sqrt{2\pi&space;\alpha&space;}}&space;-&space;\frac{1}{\delta&space;^{2}}&space;\cdot&space;\frac{1}{2}&space;\sum_{i&space;=&space;1}^{m}&space;(y^{i}&space;-&space;w^{T}&space;x^{i})&space;^{2}&space;-&space;\frac{1}{\alpha}&space;\cdot&space;\frac{1}{2}&space;w^{T}w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(w)&space;=&space;logL(w)&space;\\=mlog\frac{1}{\sqrt{2\pi&space;}\delta&space;}&space;&plus;&space;n&space;log&space;\frac{1}{\sqrt{2\pi&space;\alpha&space;}}&space;-&space;\frac{1}{\delta&space;^{2}}&space;\cdot&space;\frac{1}{2}&space;\sum_{i&space;=&space;1}^{m}&space;(y^{i}&space;-&space;w^{T}&space;x^{i})&space;^{2}&space;-&space;\frac{1}{\alpha}&space;\cdot&space;\frac{1}{2}&space;w^{T}w" title="l(w) = logL(w) \\=mlog\frac{1}{\sqrt{2\pi }\delta } + n log \frac{1}{\sqrt{2\pi \alpha }} - \frac{1}{\delta ^{2}} \cdot \frac{1}{2} \sum_{i = 1}^{m} (y^{i} - w^{T} x^{i}) ^{2} - \frac{1}{\alpha} \cdot \frac{1}{2} w^{T}w" /></a>  
    等价于：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=w_{MAP}&space;=&space;argmin(\frac{1}{\delta^{2}}&space;\cdot&space;\frac{1}{2}&space;\sum_{i&space;=&space;1}^{m}&space;(y^{i}&space;-&space;w^{T}x^{i})&space;^{2}&space;&plus;&space;\frac{1}{\alpha}&space;\cdot&space;\frac{1}{2}&space;w^{T}w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{MAP}&space;=&space;argmin(\frac{1}{\delta^{2}}&space;\cdot&space;\frac{1}{2}&space;\sum_{i&space;=&space;1}^{m}&space;(y^{i}&space;-&space;w^{T}x^{i})&space;^{2}&space;&plus;&space;\frac{1}{\alpha}&space;\cdot&space;\frac{1}{2}&space;w^{T}w)" title="w_{MAP} = argmin(\frac{1}{\delta^{2}} \cdot \frac{1}{2} \sum_{i = 1}^{m} (y^{i} - w^{T}x^{i}) ^{2} + \frac{1}{\alpha} \cdot \frac{1}{2} w^{T}w)" /></a>  
    即可等价于：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=J_{R}(w)&space;=&space;\frac{1}{n}\left&space;\|&space;y&space;-&space;w^{T}X&space;\right&space;\|^{2}&space;&plus;&space;\lambda&space;\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J_{R}(w)&space;=&space;\frac{1}{n}\left&space;\|&space;y&space;-&space;w^{T}X&space;\right&space;\|^{2}&space;&plus;&space;\lambda&space;\left&space;\|&space;w&space;\right&space;\|^{2}" title="J_{R}(w) = \frac{1}{n}\left \| y - w^{T}X \right \|^{2} + \lambda \left \| w \right \|^{2}" /></a>  
    这就是我们所说的Ridge Regression，增加了L2正则化之后的损失函数的形式。类似的，如果设定w满足拉普拉斯分布，则cost function会有一个线性项的形式。拉普拉斯分布的特殊形式会为参数带来稀疏化的优良特性。  

 `总结`
 >正则化参数等价于对参数引入先验分布，使得模型复杂度变小（缩小解空间），对于噪声以及 outliers 的鲁棒性增强（泛化能力）。整个最优化问题从贝叶斯观点来看是一种贝叶斯最大后验估计，其中正则化项对应后验估计中的先验信息，损失函数对应后验估计中的似然函数，两者的乘积即对应贝叶斯最大后验估计的形式。

从计算角度来讲，加入正则化项会加快收敛速度，越大的正则化项会使收敛速度越快。
proof:
 收敛速度取决于函数的强凸性系数，而加入了一个L2正则化项会增大正则化系数m,从而可以更快的收敛。

​    
## 优缺点分析
1. 优点
    1. 形式比较简单，可解释性较强。
    2. 作为baseline模型效果比较好，比较依赖feature engineering。
    3. 训练速度比较快，资源开销小。
2. 缺点
    1. 精度不够高，很难拟合真实的分布。
    2. 很难处理数据分布不平衡的问题。
    3. 很难处理线性不可分的问题。
    4. 本身无法筛选特征


## 多项逻辑回归
常用的多元分类的的思路是基于对问题的拆分，分为一对一拆分（OvO），一对其余拆分（OVR）和多对多（MvM）拆分。  
其中

1. OvO  
OvO将数据的N个类两两配对，，从而产生N（N-1）/2个二分类任务，最终结果可以由投票产生。一种比较简化的思路是：  

    看成是多个独立二元回归的集合：  

    实现多类别逻辑回归模型最简单的方法是，对于所有K个可能的分类结果，我们运行K−1个独立二元逻辑回归模型，在运行过程中把其中一个类别看成是主类别，然后将其它K−1个类别和我们所选择的主类别分别进行回归。通过这样的方式，如果选择结果K作为主类别的话，我们可以得到以下公式。   
    <a href="https://www.codecogs.com/eqnedit.php?latex=ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" title="ln\frac{Pr(Y_{i} = 1)}{Pr(Y_{i} = K)} = w _{} \cdot X_{i}" /></a>  
    <a href="https://www.codecogs.com/eqnedit.php?latex=ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" title="ln\frac{Pr(Y_{i} = 1)}{Pr(Y_{i} = K)} = w _{} \cdot X_{i}" /></a>
  
    ...    
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln\frac{Pr(Y_{i}&space;=&space;1)}{Pr(Y_{i}&space;=&space;K)}&space;=&space;w&space;_{}&space;\cdot&space;X_{i}" title="ln\frac{Pr(Y_{i} = 1)}{Pr(Y_{i} = K)} = w _{} \cdot X_{i}" /></a> 
  
    对上式进行指数化，有：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y_{i}&space;=&space;1)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{1}\cdot&space;X_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_{i}&space;=&space;1)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{1}\cdot&space;X_{i}}" title="P(Y_{i} = 1) = P(Y = K) e^{w_{1}\cdot X_{i}}" /></a>  
    <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y_{i}&space;=&space;2)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{2}\cdot&space;X_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_{i}&space;=&space;2)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{2}\cdot&space;X_{i}}" title="P(Y_{i} = 2) = P(Y = K) e^{w_{2}\cdot X_{i}}" /></a>  
    ...   
    <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y_{i}&space;=&space;K&space;-&space;1)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{K&space;-&space;1}\cdot&space;X_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_{i}&space;=&space;K&space;-&space;1)&space;=&space;P(Y&space;=&space;K)&space;e^{w_{K&space;-&space;1}\cdot&space;X_{i}}" title="P(Y_{i} = K - 1) = P(Y = K) e^{w_{K - 1}\cdot X_{i}}" /></a>  
    要注意的是，我们最后得到的概率和必须为1，所以我们可以得到如下的表达式：   
    假设随机变量Y的取值集合为{1,2,...,K}, 则多项逻辑回归的模型是:
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y&space;=&space;K&space;|&space;x)&space;=&space;\frac{1}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y&space;=&space;K&space;|&space;x)&space;=&space;\frac{1}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" title="P(Y = K | x) = \frac{1}{1 + \sum_{k = 1}^{K - 1}exp(w_{k} \cdot x)}" /></a>    

    <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y&space;=&space;k&space;|&space;x)&space;=&space;\frac{exp(w_{k}\cdot&space;x)}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y&space;=&space;k&space;|&space;x)&space;=&space;\frac{exp(w_{k}\cdot&space;x)}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" title="P(Y = k | x) = \frac{exp(w_{k}\cdot x)}{1 + \sum_{k = 1}^{K - 1}exp(w_{k} \cdot x)}" /></a >            
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=k&space;=&space;1,2,...,K&space;-&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k&space;=&space;1,2,...,K&space;-&space;1" title="k = 1,2,...,K - 1" /></a>

    通过这种方式，就可以扩展到多个类别的分类问题。
  
2. OvR  
OvR则是每次把一个类的样例作为正例、所有其他类的样例作为反例来训练N个分类器。测试时如果只有一个分类器预测为正例，则采用这个分类器的结果。如果多个分类器都预测为正例，则要考虑置信度区间参数。
3. MvM  
MvM是每次将若干个类作为正类，若干个类作为负类。但是类别的拆分不能随机，例如采用ECOC（纠错输出码）。

4. 看成是一个对数线性模型（softmax回归）  
    我们可以直接将其扩展成多类别回归模型。具体来说，就是使用线性预测器和额外的归一化因子（一个配分函数的对数形式）来对某个结果的概率的对数进行建模。形式如下：

    <a href="https://www.codecogs.com/eqnedit.php?latex=logPr(Y_{i}&space;=&space;1)&space;=&space;w_{1}&space;\cdot&space;X_{i}&space;-&space;logZ" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logPr(Y_{i}&space;=&space;1)&space;=&space;w_{1}&space;\cdot&space;X_{i}&space;-&space;logZ" title="logPr(Y_{i} = 1) = w_{1} \cdot X_{i} - logZ" /></a>  
    <a href="https://www.codecogs.com/eqnedit.php?latex=logPr(Y_{i}&space;=&space;2)&space;=&space;w_{2}&space;\cdot&space;X_{i}&space;-&space;logZ" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logPr(Y_{i}&space;=&space;2)&space;=&space;w_{2}&space;\cdot&space;X_{i}&space;-&space;logZ" title="logPr(Y_{i} = 2) = w_{2} \cdot X_{i} - logZ" /></a>  
    ......  
    <a href="https://www.codecogs.com/eqnedit.php?latex=logPr(Y_{i}&space;=&space;K)&space;=&space;w_{K}&space;\cdot&space;X_{i}&space;-&space;logZ" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logPr(Y_{i}&space;=&space;K)&space;=&space;w_{K}&space;\cdot&space;X_{i}&space;-&space;logZ" title="logPr(Y_{i} = K) = w_{K} \cdot X_{i} - logZ" /></a>  
    这里使用了一个额外的归一化项来使得概率可以形成一个概率分布，从而使这些概率和为1.  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{k&space;=&space;1}^{K}Pr(Y_{i}&space;=&space;k)&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{k&space;=&space;1}^{K}Pr(Y_{i}&space;=&space;k)&space;=&space;1" title="\sum_{k = 1}^{K}Pr(Y_{i} = k) = 1" /></a>  
    带入上面的求和公式，可以得到：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;\sum_{k&space;=&space;1}^{K}e^{w_{k}\cdot&space;x_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\sum_{k&space;=&space;1}^{K}e^{w_{k}\cdot&space;x_{i}}" title="Z = \sum_{k = 1}^{K}e^{w_{k}\cdot x_{i}}" /></a>  
    根据概率公式可以得到所有的概率都可以写为：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y_{i}&space;=&space;c)&space;=&space;\frac{e^{w_{c}\cdot&space;x_{i}}}{\sum_{k&space;=&space;1}^{K}e^{w_{k}\cdot&space;x_{i}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y_{i}&space;=&space;c)&space;=&space;\frac{e^{w_{c}\cdot&space;x_{i}}}{\sum_{k&space;=&space;1}^{K}e^{w_{k}\cdot&space;x_{i}}}" title="Pr(Y_{i} = c) = \frac{e^{w_{c}\cdot x_{i}}}{\sum_{k = 1}^{K}e^{w_{k}\cdot x_{i}}}" /></a>  
    这就是`softmax函数`的格式：  
    <a href="https://www.codecogs.com/eqnedit.php?latex=softmax(k,x_{1},x_{2},...,x_{n})&space;=&space;\frac{e^{x_{k}}}{\sum_{i&space;=&space;1}^{n}e^{x_{i}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?softmax(k,x_{1},x_{2},...,x_{n})&space;=&space;\frac{e^{x_{k}}}{\sum_{i&space;=&space;1}^{n}e^{x_{i}}}" title="softmax(k,x_{1},x_{2},...,x_{n}) = \frac{e^{x_{k}}}{\sum_{i = 1}^{n}e^{x_{i}}}" /></a>  
    这个函数能够将各个自变量之间的差别放大，通过softmax函数的形式可以构造出一个像是平滑函数一样的加权平均函数。 
    

注：  
    当类别标签是互斥的时候，适合用softmax回归。当类别标签不是互斥的时候，则可以使用其他的处理方式。  

## 谈一下交叉熵

我们想探讨一下为什么交叉熵用在这里来计算代价，我从两个角度来解释一下：

1.理论角度

我们要沿着熵、KL散度到交叉熵这条路线去解释：
熵表示一个事件A的自信息量，也就是A包含多少信息。KL散度可以用来表示从A的角度来看，事件B有多大的不同。而交叉熵则可以用来表示从事件A的角度来看，如何去描述事件B。一句话来总结，KL散度可以被用来计算代价，而在特定情况下最小化KL散度等价于最小化交叉熵。而交叉熵的运算最简单，所以用交叉熵来当做代价。

对于熵的定义就不多谈了，这里可以理解为表示所含信息的多少。
我们用KL散度来计算两个事件/分布之间的不同，它不具有对称性。

<a href="https://www.codecogs.com/eqnedit.php?latex=D_(KL)(A||B)&space;=&space;\sum_{i}P_A(x_i)log(P_A(x_i)&space;)&space;-&space;P_A(x_i)log(P_B(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D_(KL)(A||B)&space;=&space;\sum_{i}P_A(x_i)log(P_A(x_i)&space;)&space;-&space;P_A(x_i)log(P_B(x_i))" title="D_(KL)(A||B) = \sum_{i}P_A(x_i)log(P_A(x_i) ) - P_A(x_i)log(P_B(x_i))" /></a>

这是KL散度的定义，我们可以定义为事件A和B的差别。如果PA等于PB的话，我们可以发现，KL散度为0.而且我们还发现，减号左边就是事件A的熵，而右边是B在A上的期望。

交叉熵的公式为：

<a href="https://www.codecogs.com/eqnedit.php?latex=H(A,&space;B)&space;=&space;-\sum_{i}&space;P_A(x_i)log(P_B(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(A,&space;B)&space;=&space;-\sum_{i}&space;P_A(x_i)log(P_B(x_i))" title="H(A, B) = -\sum_{i} P_A(x_i)log(P_B(x_i))" /></a>

我们可以发现，如果A的熵值是一个常量，那么两个可以画等号，也就是说KL散度和交叉熵在特定条件下等价。

这里的A，B分别指的是真实分布和预测的分布，我们要让他们尽可能相似；所以目标就是最小化Cross entropy！

交叉熵主要是描述这两个事件相互之间的关系，对自己求交叉熵等于自己的熵值。但是交叉熵本身的表达式中，并不含有熵值这项。可以理解为，知道一个事件后对另一个事件的不确定度。

推广到ML的角度，ML在学习模型时，不能只看训练数据上的误差率和交叉熵，还要关注测试数据上的表现。所以我们的逻辑思路是，为了让学到的模型分布更贴近真实分布，我们最小化模型数据分布和训练数据之间的KL散度，而因为寻来呢数据的分布是固定的，所以最小化KL散度等价于最小化交叉熵，

所以我们用来交叉熵来计算。

2.数学推导角度
    
交叉熵的推导由似然函数而来，我们上文讲了推导。具体也可以[参考](https://blog.csdn.net/red_stone1/article/details/80735068)



