# next plan

### 数据增强

### 融合中间层

### 分层lr

### loss，增加正样本权重

# python

## `__call__()`

一般用在类中，类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以 **“对象名()”** 的形式使用，即，可调用对象。

在 python 中，实际上 `a()` 可以理解为 `a.__call__()` 的缩写，`a()` 隐式调用了`a.__call__()` 。

PS. 在神经网络中，经常可以看见 `forward()` 函数，其实也是因为 ` torch.nn.Module` （一般作为网络的父类）隐式地在 `__call()__` 中调用了 `forward()` 函数，因此将网络的输入直接传入网络的对象中，就可以实现调用 `forward()`，如 `model(x)` 。

举个例子：

```python
class Test:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 定义__call__方法
    def __call__(self,name,add):
        print("调用__call__()方法", name, age)

test = Test()
test("Jack",18)
```

输出：

```python
调用__call__()方法 Jack 18
```

---

## `()` 元组、 `[]`  列表、 `{}`  字典

- `()` 元组：有序的、**不可变的**、可索引的数据结构，可以包含任意类型的元素。**元组不可进行增删改，但是可以重新赋值。**

  ```python
  # 定义和赋值
  tuple = ()
  tuple = (1, '222', 3)
  
  # 访问元素
  ele = tuple[2]
  
  # 遍历
  for e in tuple:
      print(e)
  for e in range(len(tuple)):
      print(e)
  # 合并
  tup1 = (1, 2, 3, 4)
  tup2 = (5, 6, 7, 8)
  print(tup1 + tup2) 	# 输出：[1, 2, 3, 4, 5, 6, 7, 8]
  ```

- `[]` 列表：有序的、可变的、可索引的数据结构，可包含任意类型的元素。

  ```python
  # 定义和赋值
  list = []
  list = [1, 2, '333', [4, 5]]
  
  # 访问
  ele = list[3]
  
  # 增加元素
  list.append(6) 	# 在末尾增加新的元素
  list.extend([7, 8, [9, 10]]) 	# 在末尾增加多个元素
  list.insert(2, '222') 	# 在指定位置插入元素
  
  # 删去元素
  del list[2] 	# 删除指定索引的元素（可以是切片）
  list.pop(6) 	# 删除指定索引的元素，缺省为最后一个元素
  list.remove(6) 	# 删除指定元素的第一个匹配项
  list[7:8] = [] 	# 分片赋值实现删除
  
  # 排序
  list.sort()
  new_list = list.sorted() 	# 不改变原始列表，生成新的对象
  
  # 翻转
  list.reverse()
  
  ```

- `{}` 字典：有序的、可变的、可索引的数据结构，由键值对组成，可包含任意类型的元素。

  ```python
  # 定义和赋值
  dict = {
    'name': 'Jack',
    'age': 18
  }
  
  # 访问值
  name = dict['name']
  
  # 访问键
  key = dict.get('18', 0) 	# 键不存在，返回0
  
  # 遍历
  dict.items() 	# 将键值对按元组打包，数据类型为 dict_items
  for k, v in dict:
      print(k, v)
  
  # 插入
  dict.setdefault(key, default_key) 	# key 存在，返回对应的值；key 不存在，插入键值对
  ```

- `()` 元组 和 `[]` 列表的相互转换

  ```python
  tup = (2, 3, 7, 9)
  
  l = list(tup)
  print(l) 	# 输出[2, 3, 7, 9]
  
  t = tuple(l)
  print(t) 	# 输出(2, 3, 7, 9)
  
  # 字符串转为元组
  s = "abshwyw;123"
  print(tuple(s)) 	# 输出('a', 'b', 'y', 'w', ';', '1', '2', '3')
  ```

---

## `__getitem__()`

一般在类中定义，可以重载获取元素，让对象实现迭代功能。

- 迭代序列（列表、元组、字符串）

  ```python
  __getitem__(self, index)
  ```

- 迭代键值对（字典）

  ```
  __getitem__(self, key)
  ```

---

## `__len__() `

一般在类中定义，使得对象可以实现 `len()` 函数。（ 当执行到 **len(对象)** 方法时，会自动调用对象的 `__len__()` 方法，表示用来求该对象的某个属性（变量）的元素的个数。如果类中没有定义 `__len__()`方法，就会报类型错误。）

```
class Test(object):
    def __init__(self):
        self.a = [1, 2, 3, 4]
    def __len__(self):
        return len(self.a)
```

---

## `torch.utils.data.Dataset`

如果要自定义自己的数据集，需要定义一个继承 `torch.utils.data.Dataset` 的数据类，然后在数据类中重写 `__init__()` 和 `__len__(self) `和 `__getitem__(self, index)` 这三个方法。

重写 `__init__()` 和 `__len__() `和 `__getitem__()` 这三个方法可以自定义地读取数据，比如从 .csv 文件中读取，或是从 .txt 文件中读取数据路径，然后再在 `__getitem__()` 中返回打开的图像等操作。

```
import torch.utils.data
import torch

class DatasetTrain(torch.utils.data.Dataset): 	# 专门读取 train 的图像，可以在 __getitem__() 直接加入图像预处理
    def __init__(self, dataset_data_path, data_type='train'): 
        # 图像路径根目录
        self.img_dir =  os.path.join(dataset_data_path,"JPEGImages")     # original image dir
        self.label_dir = os.path.join(dataset_data_path,"SegmentationClass")  # mask image dir
        self.split_dir = os.path.join(dataset_data_path,"ImageSets","Segmentation")    # split_dir

        # image 初始 大小
        self.img_h = 300
        self.img_w = 300
        
        # image resize 大小
        self.new_img_h = 320
        self.new_img_w = 320
 		
        # 读取 train 的图像索引，并将相关信息存储在 examples 中
        self.examples = []
        with open(os.path.join(self.split_dir,data_type + ".txt"), "r") as f:
            lines = f.read().splitlines()
            for _, line in enumerate(lines):
                _image = os.path.join(self.img_dir,line + ".jpg")
                _mask = os.path.join(self.label_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                example = {}
                example["img_path"] = _image
                example["label_img_path"] = _mask
                example["img_id"] = line
                self.examples.append(example)
        
        self.num_examples = len(self.examples)
 
    def __getitem__(self, index): 	# 打开图像
        example = self.examples[index]
        img_id = example['img_id']
 
        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # shape: [H, W, C]
        img = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
 
        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)    # shape: [H, W]
        label_img = cv2.resize(label_img, (self.new_img_h, self.new_img_w), interpolation=cv2.INTER_NEAREST)    # shape: [H, W]
        ret, label_img = cv2.threshold(src=label_img, thresh=0, maxval=1, type=cv2.THRESH_BINARY)   # 将大于0的像素值都设为1（作为前景）
        
        ########################################################################
        # 图像预处理
        # 标准化图像
        # 转换为 tensor
        ########################################################################
        return (img, label_img, img_id)
 
    def __len__(self):
        return self.num_examples

```



---

## `torch.utils.data.DataLoader`

对数据进行 batch 的划分，按 batch 返回可迭代对象。

```python
CLASS torch.utils.data.DataLoader(dataset, 
                                  batch_size=1, 
                                  shuffle=None, 
                                  sampler=None, 
                                  batch_sampler=None, 
                                  num_workers=0, 
                                  collate_fn=None, 
                                  pin_memory=False, 
                                  drop_last=False, 
                                  timeout=0, 
                                  worker_init_fn=None, 
                                  multiprocessing_context=None, 
                                  generator=None, 
                                  *, 
                                  prefetch_factor=None, 
                                  persistent_workers=False, 
                                  pin_memory_device='')
```

<u>**parameters**</u>

> - **dataset** ([*Dataset*](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)) – 传入的数据集.
> - **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – batch 的大小 (default: `1`).
> - **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否在每个 epoch 开始时，随机排序数据 (default: `False`).
> - **num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)
> - **collate_fn** (*Callable**,* *optional*) – 处理 data load 的进程数（0 表示由主进程处理）.
> - **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 如果最后一个 batch 的小于 batch-size，设为 true 则抛弃最后这个 batch，设为 false 则不抛弃，只是最后这个 batch 会小一点 (default: `False`). 

举个例子

```
# 写在 dataset 包的 __init__.py 文件中
def gen_data_loader(args):
    if args.dataset == 'table_tennis':
        # 首先创建数据集
        train_dataset = table_tennis.DatasetTrain(dataset_data_path=os.path.join(args.data_path, args.dataset))
        val_dataset = table_tennis.DatasetVal(dataset_data_path=os.path.join(args.data_path, args.dataset))
        test_dataset = table_tennis.DatasetTest(dataset_data_path=os.path.join(args.data_path, args.dataset))
        # 然后创建 data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        return train_loader, val_loader, test_loader
    elif args.dataset == 'badminton':
        #...
```

**==ATTENTION==**

- 为什么创建训练集的 dataloader 时，一般将 shuffle 设为 true，而在验证集和测试集中，shuffle 一般设为 false？
  - 训练集的 shuffle 设为 true 的原因：
    - shuffle = True 可以使得模型 ==**在每个 epoch 开始时，重新排序数据 **== ，使得模型可以在一定 epoch 数量时，探索更多种数据分布的可能。如果不在每个 epoch 开始时重新排序数据，那么每个 epoch 对应的 batch都是相同的数据分布，由于神经网络强大的拟合能力，网络可能学习到每个 batch 的组合形式，从而影响模型的泛化能力，甚至过拟合。而在每个 epoch 开始时重新排序数据，可以减小数据顺序对模型学习的影响，提高模型的学习效率。
  - 验证集和测试集的 shuffle 设为 false 的原因：
    - 验证集的目的是选择最好的模型，测试集的目的是评估模型的泛化能力，所以验证集和测试集并不需要在每个 epoch 开始时重新排序数据，相反的，每个 epoch 的数据分布相同，才能在一次训练中评价模型每个 epoch 后的训练效果。另外，测试集的数据顺序可能与实际应用场景中的顺序一致，重新排序数据反而影响对模型在实际场景中表现的评估的准确性。

- ==**打乱两次数据集**== 通常用于机器学习中的数据处理和模型训练中。
  - 首先，在将数据分成训练集、验证集和测试集之前，通常需要对整个数据集进行第一次随机打乱。这是因为原始数据集可能具有某些固有的顺序和模式，这可能会对模型的训练和评估产生负面影响。通过随机打乱，可以消除数据集中的这些顺序和模式，使得数据更加随机和均匀地分布，从而更好地训练和评估模型。
  - 其次，训练神经网络等机器学习模型时，为了防止模型过度拟合，通常需要在每个 epoch 之前对训练集进行第二次随机打乱。这是因为如果在每个训练周期中使用相同的顺序和模式来遍历训练数据，**模型可能会记住数据的顺序和模式，从而导致过度拟合** 。通过在每个 epoch 之前对训练集进行随机打乱，可以避免这种情况的发生，提高模型的泛化能力。

---

## `torch.nn.Module`

**`nn.Module` 是所有网络结构层次的父类，当你要实现一个自己的层的时候，必须要继承这个类。**

**使用`nn.Module`的话，它就会对你神经网络的内部参数进行一个有效的管理**

## `torch.nn.Sequential`

Sequential 允许我们构建序列化的模块，也就是说用了Sequential的好处是我们可以通过数字访问第几层，可以通过parameters、weights等参数显示网络的参数和权重

## `super(myClass, self).__init__() `

1. 直接继承

   不在子类中重写 `__init__()`，默认自动调用父类的 `__init__()`

   ```
   class Father():
       def __init__(self, name='Jack'):
           self.name = name
           
   class Son(Father):
       pass
   
   son = Son()
   print(son.name) 	# 输出: Jack
   ```

2. 重写 `__init__()`

   子类中重写 `__init__()`，会 **覆盖** 父类的 `__init__()`，因此此时 name 无法被访问。

   ```
   class Father():
       def __init__(self, name='Jack'):
           self.name = name
           
   class Son(Father):
       def __init__(self, age):
           self.age = age
       pass
   
   son = Son(18)
   print(son.name) 	# 输出AttributeError: 'Son' object has no attribute 'name'
   ```

   

3. 使用 `super().__init__()`

   继承父类的 `__init__()` 的同时，还在子类中增加了新的属性。

   ```
   class Father():
       def __init__(self, name='Jack'):
           self.name = name
           
   class Son(Father):
       def __init__(self, name, age):
           super(Son, self).__init__(name)
           self.age = age
       pass
   
   son = Son('Mack', 18)
   print(son.name, son.age) 	# 输出：Mack 18
   
   ```

---

## `assert` 断言

语法：

```
assert expression

# 等价于
if not expression:
    raise AssertionError
```

```
assert expression [, arguments]

# 等价于
if not expression:
    raise AssertionError(arguments)
```

举个例子

```
assert embedding_dim % num_heads == 0, 'embedding_dim % num_heads != 0' 	
# 输出：AssertionError: embedding_dim % num_heads != 0
```

---

## `zip()`

Python 中的一个内置函数，用于将多个可迭代对象打包成一个元组序列。它的语法结构如下：

如果使用 `zip()` 函数将它们打包成一个元组序列，其中每个元素都是一个元组，包含 `a` 和 `b` 对应位置的元素：

```pythob
a = [1, 2, 3]
b = [4, 5, 6]

result = zip(a, b)
```

输出

```python
[(1, 4), (2, 5), (3, 6)]
```

---

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

## `torch.squeeze()`

降维，但是智能压缩维数为 1 的维度。

```python
x = torch.zeros(2, 1, 2, 1, 2)
x.size() 	# torch.Size([2, 1, 2, 1, 2])

y = torch.squeeze(x)         # 压缩维数为 1 的维度
y.size() 	# torch.Size([2, 2, 2])
```

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



