# Logistic Regression
## 推导逻辑

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
如果把LR用于多分类，假设随机变量Y的取值集合为{1,2,...,K}, 则多项逻辑回归的模型是:  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(Y&space;=&space;k&space;|&space;x)&space;=&space;\frac{exp(w_{k}\cdot&space;x)}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y&space;=&space;k&space;|&space;x)&space;=&space;\frac{exp(w_{k}\cdot&space;x)}{1&space;&plus;&space;\sum_{k&space;=&space;1}^{K&space;-&space;1}exp(w_{k}&space;\cdot&space;x)}" title="P(Y = k | x) = \frac{exp(w_{k}\cdot x)}{1 + \sum_{k = 1}^{K - 1}exp(w_{k} \cdot x)}" /></a>