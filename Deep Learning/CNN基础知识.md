- Batch Normalization：

机器学习领域有个很重要的假设：假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。那BatchNorm的作用是什么呢？BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

BN的基本思想其实相当直观：因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，这是训练深层神经网络收敛越来越慢的本质原因，而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。

对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。

用一句话来总结，其实BN就是把隐层神经元激活输入从不拘一格的输入，变成均值是0、方差为1的正态分布。经过BN之后，大部分Activation的值会落入非线性函数的线性区，使其远离饱和区，这样可以加速训练收敛速度。

prediction中不需要bn层。

![BN](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/BN.png)

BN层是在激活层之前加入，对于mini-batch的SGD，我们的变换公式就可以写成：

<a href="https://www.codecogs.com/eqnedit.php?latex=x^{(k)}&space;=&space;\frac{x^{(k)}&space;-&space;E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{(k)}&space;=&space;\frac{x^{(k)}&space;-&space;E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}" title="x^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}" /></a>

如果只进行这样的变换会带来网络表达能力的下降，所以我们加入两个参数，这两个参数是通过学习得到：（因为它意味只利用了激活函数的线性部分，可能造成表达能力的下降。）

<a href="https://www.codecogs.com/eqnedit.php?latex=y^{(k)}&space;=&space;\gamma&space;^{(k)}x^{(k)}&space;&plus;&space;\beta^{(k)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^{(k)}&space;=&space;\gamma&space;^{(k)}x^{(k)}&space;&plus;&space;\beta^{(k)}" title="y^{(k)} = \gamma ^{(k)}x^{(k)} + \beta^{(k)}" /></a>

所以BN为了保证非线性的获得，对变换后的满足均值为0方差为1的x又进行了scale加上shift操作(y=scale*x+shift)，每个神经元增加了两个参数scale和shift参数，这两个参数是通过训练学习到的，意思是通过scale和shift把这个值从标准正态分布左移或者右移一点并长胖一点或者变瘦一点，每个实例挪动的程度不一样，这样等价于非线性函数的值从正中心周围的线性区往非线性区动了动。核心思想应该是想找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头使得网络收敛速度太慢。

[参考文献](https://www.cnblogs.com/guoyaohua/p/8724433.html)
