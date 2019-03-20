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

给定一个学习任务, 在训练初期, 由于训练不足, 学习器的拟合能力不够强, 偏差比较大, 也是由于拟合能力不强, 数据集的扰动也无法使学习器产生显著变化, 也就是欠拟合的情况; 随着训练程度的加深, 学习器的拟合能力逐渐增强, 训练数据的扰动也能够渐渐被学习器学到; 充分训练后, 学习器的拟合能力已非常强, 训练数据的轻微扰动都会导致学习器发生显著变化, 当训练数据自身的、非全局的特性被学习器学到了, 则将发生过拟合.
## How to Solve

通过train accuracy 和 test accuracy的关系来判断当前模型的状态：

或者通过train loss和test loss来判断：
如果train loss和test loss同时还都在减小的阶段，则大概率他们处于过拟合；
如果train loss处于平稳，而test loss在升高，这时则是处于欠拟合。


solve underfit:
1. more features
2. more epochs
3. complex models(nonlinear model)
4. less regularization

solve overfit:
1. cross validation
2. regularization
3. less features
4. more data
5. Bagging

In DNN:

1. early stop
2. Dropout
3. BatchNormailzation(主要是用来加速网络收敛，但是通过一个batch内的数据实现相互关联，使得同一个样本的输出不仅仅依赖与一个输入。这也是一种数据增强，一个样本在超平面上被拉扯，每次拉扯的方向不同。这种数据增强的是贯穿整个神经网络的。)
4. data augumentation（more data）
5. Regularization（权重衰减）
6. less complex NN 
7. Add Noise(输入中，权重中，输出中)
8. Boosting 和 Bagging为主的集成模型

