# Variance and bias

## 误差来源

<a href="https://www.codecogs.com/eqnedit.php?latex=E[(y&space;-&space;\hat{f(x)})^2]&space;=&space;E[(y&space;-&space;E[y]^2)]&space;&plus;&space;(E[y]&space;-&space;E[\hat{f(x)}])^2&space;&plus;&space;E[(E[\hat{f(x)}&space;-&space;\hat{f(x)}])^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[(y&space;-&space;\hat{f(x)})^2]&space;=&space;E[(y&space;-&space;E[y]^2)]&space;&plus;&space;(E[y]&space;-&space;E[\hat{f(x)}])^2&space;&plus;&space;E[(E[\hat{f(x)}&space;-&space;\hat{f(x)}])^2]" title="E[(y - \hat{f(x)})^2] = E[(y - E[y]^2)] + (E[y] - E[\hat{f(x)}])^2 + E[(E[\hat{f(x)} - \hat{f(x)}])^2]" /></a>

这里第一项被称为noise，第二项是bias，第三项是variance。
![bias](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/variances_bias.png)

![bias2](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/var_bias2.png)

bias: 偏差度量了学习算法的期望预测与真实结果的偏离程序, 即 刻画了学习算法本身的拟合能力 .
variances:  方差度量了同样大小的训练集的变动所导致的学习性能的变化, 即 刻画了数据扰动所造成的影响 .

noise: 噪声表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界, 即 刻画了学习问题本身的难度 . 巧妇难为无米之炊, 给一堆很差的食材, 要想做出一顿美味, 肯定是很有难度的.


## Application in ML and DL

欠拟合，对应偏差高。显然，欠拟合就是本身拟合训练数据都不行，也就是训练误差也高。预测的值离真实值的距离就偏高。用模型复杂度来说，就是模型复杂度不够。

过拟合，对应方差高。也就是训练得到的模型太拟合训练数据了。不同的训练数据训练的模型效果波动很大。泛化能力弱。用模型复杂度来说，就是模型太复杂了。

## How to Solve

solve underfit:


solve overfit:


