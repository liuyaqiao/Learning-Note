# Logistic Regression
## 推导逻辑

前提：LR是一种使用线性组合的形式来逼近数据的分布，可以使用wx的形式来表示数据的分布。

LR遵从的是的是伯努利分布，伯努利分布的概率表达式为:   
  
伯努利分布是一个离散的两点分布，结果只有是或否两种情况。  
  
对于随机变量而言，有如下的概率：  
  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[x&space;=&space;1]&space;=&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[x&space;=&space;1]&space;=&space;p" title="Pr[x = 1] = p" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr[x&space;=&space;0]&space;=&space;1&space;-&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr[x&space;=&space;0]&space;=&space;1&space;-&space;p" title="Pr[x = 0] = 1 - p" /></a>

 我们考虑事件发生的对数几率：（事件发生与不发生的比值）  
 <a href="https://www.codecogs.com/eqnedit.php?latex=logit&space;=&space;log(\frac{p}{1-p})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?logit&space;=&space;log(\frac{p}{1-p})" title="logit = log(\frac{p}{1-p})" /></a>  
我们可以采用线性组合来表示对数几率（依据上文）：  
<a href="https://www.codecogs.com/eqnedit.php?latex=log(\frac{p}{1-p})&space;=&space;wx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(\frac{p}{1-p})&space;=&space;wx" title="log(\frac{p}{1-p}) = wx" /></a>  
通过上面的表达式我们可以求出：  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;1|x)&space;=&space;\frac{e^{wx}}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 1|x) = \frac{e^{wx}}{e^{wx} + 1}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr(Y&space;=&space;0|x)&space;=&space;\frac{1}{e^{wx}&space;&plus;&space;1}" title="Pr(Y = 0|x) = \frac{1}{e^{wx} + 1}" /></a>  
上面两个表达式的大小决定的测试例的归属，所以我们为了更方便的判断，我们选择了sigmoid函数。这里考虑到了sigmoid对称性好、值域在[0,1]，并且具有对称性。  
sigmoid函数为：
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=\frac{1}{1&space;&plus;&space;e^{-z}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=\frac{1}{1&space;&plus;&space;e^{-z}}" title="f(x) =\frac{1}{1 + e^{-z}}" /></a>



