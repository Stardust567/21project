## 占位符

```python
x = tf.placeholder(tf.float32, [None, 784])
```

tf的占位符，用于传入外部数据

> placeholder(dtype, shape=None, name=None):
>
> :param dtype: 数据类型
> :param shape: 数据维度，None表无限制
> :param name: 名称
> :return: Tensor类型

例子：TensorFlow中加载图片的维度为[batch, height, width, channels]

故placeholder的shape可写为[None, None, None, 3]

## 变量

```python
W = tf.Variable(tf.zeros([784, 10]))
```

tf变量需要初始值，一旦初始值确定，那么该变量的类型和形状就基本确定了

> Variable(initial_value=None, trainable=True, validate_shape=True, name=None):
>
> :param initial_value:初始值，可以搭配tensorflow随机生成函数
> :param trainable:默认该变量可被算法优化，不想该变量被优化，改为False
> :param validate_shape:默认形状不接受更改，如需更改，改为False
> :param name:给变量确定名称
> :return: Tensor类型

## 会话

```python
sess = tf.InteractiveSession()
```

对上述结点进行计算的上下文，变量的值会保存在会话中。

## softmax

即归一化指数函数，将多分类中各个类别的评价分数转换为和为1的分布概率

![{\displaystyle \sigma (\mathbf {z} )_{j}={\frac {e^{z_{j}}}{\sum _{k=1}^{K}e^{z_{k}}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)    for *j* = 1, …, *K*.

## 交叉熵

### 信息量

一件事发生的可能性越小，其发生所带来的信息量越大；反之，一件发生概率为1的事发生，对我们来说毫无价值，获取到的信息量为0。

假设X是一个离散型随机变量，定义事件X=x0的信息量为：I(x0)=−log(p(x0))，其中p(x0)取值为[0, 1]

### 熵

对于某个事件，熵反映了其所有可能性所带来的信息量，是一个随机变量的确定性的度量。熵越大，变量的取值越不确定，反之就越确定。

熵用来表示所有信息量的期望H(X)=−∑p(xi)*log(p(xi))

### 相对熵

即KL散度，如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度来衡量这两个分布的差异，即如果用P来描述目标问题，而不是用Q来描述目标问题，得到的信息增量。

在机器学习中，P表示样本真实分布，比如[1,0,0]（样本属于第一类）Q表示模型预测分布，比如[0.7,0.2,0.1] 

对于离散随机变量，其概率分布P 和 Q的KL散度可按下式定义为D~KL~(P|Q) = ∑~i~P(x~i~)log(p(x~i~)/q(x~i~))其中D~KL~的值越小，表示q分布和p分布越接近。

### 交叉熵

相对熵中D~KL~(P|Q) =  ∑~i~P(x~i~)log[p(x~i~)/q(x~i~)] = ∑~i~P(x~i~)log(p(x~i~)) - ∑~i~P(x~i~)log(q(x~i~)) = -H(p(x)) + [-∑~i~P(x~i~)*log(q(x~i~))]

其中真实样本熵固定，所以直接用-∑~i~P(x~i~)*log(q(x~i~))作为优化指标（交叉熵）即可，即交叉熵越小越好。

#### 交叉熵VS均方根误差

做分类问题时，为什么我们不用RSME而是交叉熵作损失函数呢？

1. 我们希望损失函数能做到，当预测值跟目标值越远时，修改参数后能减去一个更大的值，做到更加快速的下降。

2. 函数更不容易陷入局部最优解。

在做后向传播时，会出现（基于一个样本的情况，对于softmax下w，b偏导进行计算）：

1. RSME在更新w，b时候，w,b的梯度跟激活函数的梯度成正比，激活函数梯度越大，w,b调整就越快，训练收敛就越快，但是Simoid函数在值非常高时候，梯度是很小的，比较平缓。

2. 交叉熵在更新w,b时候，w,b的梯度跟激活函数的梯度没有关系了，bz已经表抵消掉了，其中bz-y表示的是预测值跟实际值差距，如果差距越大，那么w,b调整就越快，收敛就越快。

#### 均方根误差VS平均绝对误差

而在做回归问题时，我们的选择又该是什么样呢？

RSME：若出现误差较大的点，RSME将被调整以最小化这个离群数据点，但却是以牺牲其他正常数据点的预测效果为代价，这最终会降低模型的整体性能。

MAE：若出现误差较大的点，最小化MAE的预测为所有目标值的中位数。我们知道中位数对于离群点比平均值更鲁棒，这使得MAE比MSE更加鲁棒。但使用MAE损失（特别是对于神经网络）的一个大问题是它的梯度始终是相同的，这意味着即使对于小的损失值，其梯度也是大的。

#### Log-Cosh Loss

Log-cosh是用于回归任务的另一种损失函数，它比之前两种更加平滑。顾名思义，它采用 ∑~i~log(cosh(y~i~^p^-y~i~))作预测误差。Log-cosh Loss对于小的x来说，其大约等于 (x ** 2) / 2，而对于大的x来说，其大约等于 abs(x) - log(2)。这意味着logcosh的作用大部分与均方误差一样，但不会受到偶尔出现的极端不正确预测的强烈影响，**且二阶可导**。



