# python



## `parser` `store_true`

在 `argparse` 中，`add_argument()` 方法的 `action` 参数可以设置为 `store_true`，表示如果命令行中指定了该参数，则将其存储为 `True`，否则存储为 `False`。这通常用于解析布尔类型的命令行参数，例如：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='increase output verbosity')
args = parser.parse_args()

if args.verbose:
    print('Verbose output enabled')
else:
    print('Verbose output disabled')
```

在上面的例子中，`add_argument()` 方法使用 `action='store_true'` 参数将命令行参数解析为布尔类型。如果命令行中指定了参数 `--verbose`，则将其存储为 `True`，否则存储为 `False`。然后，根据命令行参数的值，打印不同的输出。

如果在命令行中执行以下命令：

```python
python script.py --verbose
```

那么将会输出 `Verbose output enabled`。

如果不指定命令行参数 `--verbose`，则输出为 `Verbose output disabled`。

---

## `torch.max` & `torch.argmax` & `np.argmax` & `np.max`

- `np.argmax(np.array, axis)`

  在数组的第 axis 轴上求最大值，返回数组中最大值的索引值。当同时出现多个最大值时，返回第一个最大值的索引。

  举个例子：

  ```python
  import numpy as np
  a = np.array([
      [
          [1, 5, 5, 2],
          [9, -6, 2, 8],
          [-3, 7, -9, 1]
      ],
  
      [
          [-1, 7, -5, 2],
          [9, 6, 2, 8],
          [3, 7, 9, 1]
      ],
  
      [
          [21, 6, -5, 2],
          [9, 36, 2, 8],
          [3, 7, 79, 1]
      ]
  ])
  
  b = np.argmax(a, axis = 0) 	# 在第 0 维上比较，也就是 1，-1,21
  print(b)
  # 输出
  '''
  	[[2 1 0 0]
       [0 2 0 0]
       [1 0 2 0]]
  '''
  c = np.argmax(a, axis = 1) 	# 在第 1 维上比较，也就是 1,9，-3
  print(c)
  # 输出
  '''
  	[[1 2 0 1]
       [1 0 2 1]
       [0 1 2 1]]
  '''
  d = np.argmax(a, axis = 2) 	# 在第 2 维上比较，也就是 1,5,5,2
  print(d)
  # 输出
  '''
  	[[1 0 1]
       [1 0 2]
       [0 1 2]]
  '''
  ```

- `torch.argmax(input, dim)`

  不指定 dim 时，默认将输入张量 input 展开成一维张量，然后返回最大值的下标。指定 dim 时，在数组的第 dim 轴上求最大值，返回数组中最大值的索引值。当同时出现多个最大值时，返回第一个最大值的索引。

  ==常用在分类任务中，筛选出网络预测值 pred 中最大概率值对应的 class index。== 

- `np.max(np.array, axis)`

  不指定 axis 时，默认将输入数组 np.array 展开成一维张量，然后返回最大值。在数组的第 axis 轴上求最大值，返回数组中的最大值。

  ```
  import numpy as np
  a = np.array([
      [
          [1, 5, 5, 2],
          [9, -6, 2, 8],
          [-3, 7, -9, 1]
      ],
  
      [
          [-1, 7, -5, 2],
          [9, 6, 2, 8],
          [3, 7, 9, 1]
      ],
  
      [
          [21, 6, -5, 2],
          [9, 36, 2, 8],
          [3, 7, 79, 1]
      ]
  ])
  
  print(np.max(a, axis=0))
  # 输出
  '''
      [[21  7  5  2]
       [ 9 36  2  8]
       [ 3  7 79  1]]
  '''
  ```

- `torch.max(input, dim)`

  会返回两个张量，一个是最大值，一个市最大值对应的索引。不指定 dim 时，默认将输入张量 input 展开成一维张量，然后返回最大值以及下标。指定 dim 时，在数组的第 dim 轴上求最大值，返回数组中最大值以及索引值。当同时出现多个最大值时，返回第一个最大值以及索引。

  ```
  import torch
  
  a = torch.tensor([
      [
          [1, 5, 5, 2],
          [9, -6, 2, 8],
          [-3, 7, -9, 1]
      ],
  
      [
          [-1, 7, -5, 2],
          [9, 6, 2, 8],
          [3, 7, 9, 1]
      ],
  
      [
          [21, 6, -5, 2],
          [9, 36, 2, 8],
          [3, 7, 79, 1]
      ]
  ])
  print(torch.max(a, dim=0))
  # 输出
  '''
      torch.return_types.max(
      values=tensor([[21,  7,  5,  2],
                      [ 9, 36,  2,  8],
                      [ 3,  7, 79,  1]]),
      indices=tensor([[2, 1, 0, 0],
                      [0, 2, 0, 0],
                      [1, 0, 2, 0]]))
  '''
  ```

  

---

## ==我的 loss 是怎么改出来的？？？？？ tensor variable 梯度问题==



## [Pytorch训练模型损失Loss为Nan或者无穷大（INF）原因_loss为inf_ytusdc的博客-CSDN博客](https://blog.csdn.net/ytusdc/article/details/122321907?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-122321907-blog-105880368.235%5Ev29%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-122321907-blog-105880368.235%5Ev29%5Epc_relevant_default_base3&utm_relevant_index=2)



## `import sys`

### 0. `sys.path` 和 `sys.path.append`

`sys.path` 是一个列表。

在 python 文件中使用 import 后，python 解释器默认会搜索当前目录、已安装的内置模块和第三方模块，儿子而这些搜索到的路径就会被保存在 `sys.path` 这个列表中。

若要添加自己的导入模块（比如导入另一个项目中的库），可以通过 `sys.path.append` 在 `sys.path` 列表中加入路径，从而使得 python 解释器可以搜索到要导入的模块。



## python 中的 *

- 乘号
- 收集列表中多余的值
- 函数的形参
  - 单星号，接受列表
  - 双星号，接受字典

[参考博客](https://blog.csdn.net/zkk9527/article/details/88675129?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168308736716800197079692%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=168308736716800197079692&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-15-88675129-null-null.142^v86^insert_down28,239^v2^insert_chatgpt&utm_term=python%20def%20%28*%29&spm=1018.2226.3001.4187)

## <mark>map()</mark>

dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))

## `torch.arange`

![image-20230504193947163](C:\Users\ly\AppData\Roaming\Typora\typora-user-images\image-20230504193947163.png)

```
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

> **Parameters**：
>
> - **start** - 起始值，缺省为 0。
> - **end** - 结束值。
> - **step** - 步长，缺省为 1。

**举个例子**

```
torch.arange(5)  # tensor([ 0,  1,  2,  3,  4])
```

---



## `torch.chunk`

将 tensor 切分成给定数量的切片。

```
torch.chunk(input, chunks, dim=0) → List of Tensors
```

> Parameters：
>
> - **input** - 要切分的 tensor
> - **chunks** - 要切分的数量。如果给定的维度 dim **可以整除** chunks，那么得到的每个切片大小都相同；但如果**无法整除**，则除了最后一个切片以外，其他切片的大小相同。
> - **dim** - 在 dim 维度上进行切分

**举个例子**

```
torch.arange(11).chunk(6)
'''
(tensor([0, 1]),
 tensor([2, 3]),
 tensor([4, 5]),
 tensor([6, 7]),
 tensor([8, 9]),
 tensor([10]))
'''
```

---



## `torch.einsum`

爱因斯坦求和约定（einsum）提供了一套既简洁又优雅的规则，可实现包括但不限于：向量内积，向量外积，矩阵乘法，转置和张量收缩等张量操作，熟练运用 einsum 可以很方便的实现复杂的张量操作。

```
torch.einsum(equation, *operands) → Tensor
```

> **Parameters**：
>
> - **equation** - 运算规则
> - **operands** - 进行运算的张量

**举个例子**

```
# batch matrix multiplication 以 batch 为单位进行矩阵乘法
As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)
torch.einsum('bij,bjk->bik', As, Bs)
'''
tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
        [-1.6706, -0.8097, -0.8025, -2.1183]],

        [[ 4.2239,  0.3107, -0.5756, -0.2354],
        [-1.4558, -0.3460,  1.5087, -0.8530]],

        [[ 2.8153,  1.8787, -4.3839, -1.2112],
        [ 0.3728, -2.1131,  0.0921,  0.8305]]])
'''
# transpose the lastest two dims 对最后两维做转置
b = torch.einsum('...ij->...ji', [a])
```



## `torch.cat`

拼接 tensors（tensor 必须是 **相同尺寸** 的）

```
torch.cat(tensors, dim=0, *, out=None) → Tensor
```

> **Parameters**：
>
> - **tensors** - 要拼接的 tensor。
> - **dim** - 在 dim 维度上进行拼接。

**举个例子**

```
x = torch.randn(2, 3)
torch.cat((x, x, x), 0) 	# [6, 3]
torch.cat((x, x, x), 1) 	# [2, 9]
```

---



## <mark>contiguous()</mark>

https://blog.csdn.net/kdongyi/article/details/108180250

## <mark>enumerate</mark>

## <mark>nn.Upsample</mark>

## <mark>torch.arange(self.seq_length).expand((1, -1))</mark>



## `self.register_buffer`

将不需要更新的参数保存到 `model.state_dict()` 中。

比如你像这样想使用一个类内成员，

```python
class block(nn.Module):
    def __init__(self):
        self.tensor = torch.rand(1, 3, 224, 224)
    def forward(self, x):
        return x + self.tensor
```

由于 `self.tensor` 是在 cpu 中的，所以无法直接进行计算。这时候可以将 `self.tensor` 通过 `register_buffer` 放入到 `model.state_dict()`，这样 `self.tensor` 就会随着 `model.cuda()` ，被复制到 gpu 上。像这样：

```
class block(nn.Module):
    def __init__(self):
        self.register_buffer('my_tensor', torch.rand(1, 3, 224, 224)) 
    def forward(self, x):
        return x + self.my_tensor
```

PS. 模型保存下来的参数有两种：一种是需要更新的 Parameter，另一种是不需要更新的 buffer 。在模型中，利用 backward 反向传播，可以通过 `requires_grad` 来得到 buffer 和 parameter 的梯度信息，但是利用 optimizer 进行更新的是 parameter ，buffer 不会更新，这也是两者最重要的区别。这两种参数都存在于`model.state_dict()`的 `OrderedDict` 中，也会随着模型“移动”（`model.cuda()`）。[参考博文](https://blog.csdn.net/weixin_46197934/article/details/119518497)

---



# Transformer

## 解决痛点：

RNN，训练慢

## input embedding

RGB [C, H, W] 首先在一个小 backbone(在ImageNet上预训练过) 上进行 input embedding，得到 [C', H', W']

## flatten

[C', H', W'] --> [H' x W', C']

## positional encoding

## encoder block

### attention

[H' x W', 3] 作为 Q, K, V 的输入，输出也是 [H' x W', 3]

将 feature 映射到 3 个不同的高维空间（latent space）

<mark>Q · KT ：代表每个像素和其他像素的关系的强弱</mark>，得到 [H' x W', H' x W']

### 为什么要切 patch？

因为不切的话，经过softmax后，值被多个像素均分了，网络难以学习强弱关系。（sparse）

再 dot V：每个像素和其他像素的关系 与 V 相乘，类似于加权聚合 aggregate。

## multi-head

[H' x W', C'] 经过 n 个头，然后 concat，得到[H' x W', C', n]，再做 1x1 卷积降维，得到[H' x W', C']

# ViT

每张图像 = 多个 16x16 的 words



## <mark>mobileViT</mark>

### transformer的痛点

- 参数量大，算力要求高
- 缺少空间归纳偏置
- 迁移到其他任务比较繁琐（位置编码约束输入分辨率，一般用插值，但是会掉点）
- 模型训练困难（更大的数据集，更大的epoch才能收敛，更多的数据增强（对数据增强敏感，去掉会掉很多点））

ps.模型参数量少不代表推理时间短

mobileViT 

- 对数据增强没有那么敏感

ViT(无class token)

![image-20230504123012637](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230504123012637.png)



mobileViT

![image-20230504123130355](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230504123130355.png)





# 先做上采样，或者别的操作，强化形状和颜色特征，然后做卷积，再做transformer，然后做工程上的加速。

模板匹配

# INK

## 0. work相关

做完 CS231n 所有作业，面试时基础知识是没有问题的

CSE 599w - system for ML homework（可以写到简历上）

基本算子实现：conv、attention算子 + python、c++实现 + 不同硬件的programming model加速（intel oneDNN）

paper最新进展

尝试称为开源框架的contributor：DGL，MM，Paddle，Tensorflow，Pytorch

研究生校招：（后续根据工作能力和对公司的贡献晋升）

华为 - 13级，好一点14级

百度 - T3

绩效前20% - 年终奖

## 1. 背景描述

输入：.tif文件。3d信息，包含深度，深度是在 z 方向是的65个灰度切片



## 2. edge loss（介于目标检测和语义分割之间）

对于语义不可知的/困难的特征，如何使用 edge loss 实现任务。



## 3.



