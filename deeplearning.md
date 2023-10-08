# python



## numpy

```
gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
```



## glob



### `glob.glob(pathname, (optional) recursive=False)`：搜索指定格式文件名

参数：

- `pathname`：包含通配符的路径模式，用于指定要搜索的文件和目录的规则
- `recursive`：可选参数，若 `recursive = true`，则会递归搜索子目录。

输出：

返回一个字符串列表，每个字符串代表一个匹配的文件或目录的完整路径

e.g.

```
import glob

files = glob.glob("/path/to/directory/*") 	# 返回该目录下所有文件的列表
txt_files = glob.glob("/path/to/directory/*.txt") 	# 返回该目录下所有 txt 文件的列表
all_files = glob.glob("/path/to/directory/**/*", recursive=True)
```

[golb.glob]: https://blog.csdn.net/shary_cao/article/details/122050756?ops_request_misc=&amp;request_id=&amp;biz_id=102&amp;utm_term=glob.glob&amp;utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-122050756.142^v95^insert_down1&amp;spm=1018.2226.3001.4187	"golb.glob"

---



# pytorch

## torch

### `torch.from_numpy(array)`： 将 np 转换为 tensor

```
import numpy as np
import torch
inp = np.random.randint(0, 256, size=(1920, 1200, 3), dtype=np.uint8) 	# 随机生成一个 1920x1200x3 的列表
inp = torch.from_numpy(inp).float().permute((2, 0, 1)).unsqueeze(dim = 0) 	# 将列表转换为 tensor，再转换为 float（否则内部卷积时类型会不匹配），再交换维度，再增加一个维度。
```

### `torch.permute((2, 0, 1))`：重新排序维度

```python
import numpy as np
import torch
inp = np.random.randint(0, 256, size=(1920, 1200, 3), dtype=np.uint8) 	# 随机生成一个 1920x1200x3 的列表
inp = torch.from_numpy(inp).float().permute((2, 0, 1)).unsqueeze(dim = 0) 	# 将列表转换为 tensor，再转换为 float（否则内部卷积时类型会不匹配），再交换维度，再增加一个维度。
```

### `torch.unsqueeze(dim = i)`：在第 i 维增加一个为 1 的维度

```
import numpy as np
import torch
inp = np.random.randint(0, 256, size=(1920, 1200, 3), dtype=np.uint8) 	# 随机生成一个 1920x1200x3 的列表
inp = torch.from_numpy(inp).float().permute((2, 0, 1)).unsqueeze(dim = 0) 	# 将列表转换为 tensor，再转换为 float（否则内部卷积时类型会不匹配），再交换维度，再增加一个维度。
```

### `torch.cuda.is_available()`：检查当前系统是否可用 CUDA

```
self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
# 如果 self.cuda 为 True，则将设备设置为第一个可用的 CUDA 设备（通常命名为 "cuda:0"），否则将设备设置为 CPU。这个设备将用于分配张量和执行计算操作。
```

### `torch.device()`：指定 pytorch 设备

torch.to(device)

```
self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
# 如果 self.cuda 为 True，则将设备设置为第一个可用的 CUDA 设备（通常命名为 "cuda:0"），否则将设备设置为 CPU。这个设备将用于分配张量和执行计算操作。

self.net = self.net.to(self.device)
```

### 初始化模型权重

```
def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

### 导入预训练权重

```
load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
```



### net.parameters(),

### optim.SGD

```
self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
```



### `self.lr_scheduler = ExponentialLR(self.optimizer, gamma)`