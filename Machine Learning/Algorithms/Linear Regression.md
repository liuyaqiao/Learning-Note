# Linear Regression

## Assumption

假设就是一个线性假设，y 可以表示成参数beta和x的线性组合。

这里要注意，在feature数值化的时候不能表示为数值，需要表示成Categorical的形式；这样可以形成许多稀疏矩阵，可以带来存储空间和计算速度上的好处；

## Loss

Least Squares Estimator: 找出一组beta，使得LSE最小。
![lse](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/LSE_beta.png)


## Noise

如果加入noise，则会和MLE分布估计类似：

1. Gaussian Noise
  ![gaussnoise](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/Gaussian_noise.png)
  所以有：
  最小化误差函数，就相当于最大化对数似然函数。这等同于最大似然估计；

2. Laplace Noise
  同理，只是此时最大化的是绝对值误差函数。

注意：

这里和Regularization是为什么加入L1，L2项有一定的共同之处，都是我们提前约束了参数的分布，L1正则化我们规定参数满足拉普拉斯分布，而L2正则化我们规定参数满足高斯分布，所以在推导项中，最后的误差结论中，分别有一个绝对值项和平方项。

## Regularization