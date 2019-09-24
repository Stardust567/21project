### CNN

```python
x_image = tf.reshape(x, [-1, 28, 28, 1])
```

因为cnn需要在图片的像素矩阵上进行池化等操作，所以需要将原来的784\*1向量转成28\*28的矩阵（[-1, 28, 28, 1]中的-1形状的第一维大小是根据x自动确定的）

#### tf.nn.conv2d()

```python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```

 `tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)`

input：需要做卷积的输入图像(Tensor=[batch, in_height, in_width, in_channels])即[训练时一个batch的图片数量, 图片高, 图片宽, 图像通道数]，该Tensor要求类型为float32或float64

filter：CNN卷积核(Tensor=[filter_height, filter_width, in_channels, out_channels])即[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型**与参数input相同**，有一个地方需要注意，**第三维in_channels，就是参数input的第四维**

strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

padding：只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式

use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true

return：Tensor，就是我们常说的feature map

#### tf.nn.max_pool()

```python
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

`tf.nn.max_pool(value, ksize, strides, padding, name=None)`

value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape

ksize：池化窗口的大小，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

padding：和卷积类似，可以取'VALID' 或者'SAME'

return：Tensor=[batch, height, width, channels]