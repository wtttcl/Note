# gcc

GCC 原名 GNU C 语言编译器（GNU C Compiler），只能处理C语言。但其很快扩展，变得可处理C++，后来又扩展为能够支持更多编程语言，如Pascal、Objective -C、Java、Go以及各类处理器架构上的汇编语言等，所以改名GNU编译器套件（GNU Compiler Collection）。



## gcc 和 g++ 的区别

- gcc 和 g++ 都是 GNU 的 **编译器**；
- gcc 和 g++ 都可以编译 c 代码和 c++ 代码。但是：
  - 后缀为 .c 的程序，**gcc 认为其是 c 程序， g++ 认为其是 c++ 程序**；
  - 后缀为 .cpp 的程序，gcc 和 g++ 都认为其是 c++ 程序；
  - 编译阶段，g++ 会调用 gcc。所以对于 c++ 代码来说，gcc 和 g++ 是等价的；但是因为 **gcc 不能自动和 c++ 程序使用的库链接，需要 g++ 来完成链接**，所以为了统一起见，干脆 **编译和链接 c++ 程序都使用 g++** 了，但实际上，**编译调用的是 gcc**。
- 编译可以使用 gcc / g++，链接可以使用 g++ / gcc -lstdc++（用 c++ 的标准进行链接）。gcc 不能自动和 c++ 程序使用的库链接，需要 g++ 来完成链接；g++ 编译的时候会调用 gcc。
- `__cpluscplus`：宏。标志着编译器将代码当做 c 代码来解释还是当作 c++ 代码来解释。
  - 如果程序后缀为 .c，并且采用 gcc 编译器，那么编译器认为这是 c 代码，所以该宏是未定义的；
  - 如果编译器认为这是 c++ 代码，那么该宏就可以被定义。



## gcc 使用

<img src="./assets/e8e38f8119ccdc53eaf3ec57c108dff.jpg" alt="e8e38f8119ccdc53eaf3ec57c108dff" style="zoom: 45%;" />

<img src="./assets/c46f3622f0908474c6f31f891628626.jpg" alt="c46f3622f0908474c6f31f891628626" style="zoom:45%;" />

```shell
# gcc file_name.c -o target_name 将 file_name.c 编译成可执行文件（可执行文件名为 target_name）
gcc test.c -o test

# gcc 要生成 test 可执行文件，需要用到 test.c 文件
gcc -o test test.c

# gcc file_name.c 将 file_name.c 编译成可执行文件（默认可执行文件名为 a.out）
gcc filename.c

# 运行可执行文件
./target_name
./a.out 	# linux 默认的可执行文件名
```



<img src="./assets/03cb0d48c26a96a88c4d6141f65ae4a.jpg" alt="03cb0d48c26a96a88c4d6141f65ae4a" style="zoom: 35%;" />

- 预处理后源代码：删去注释、宏替换等

```
gcc test.c -E -o test.i 	# 生成预处理后文件 test.i， -E 预处理 -o 生成目标文件

gcc test.i -S -o test.s 	# 生成编译后文件 test.s， -S 编译 -o 生成目标文件

gcc test.s -c -o test.o 	# 生成汇编后文件 test.o， -S 编译 -o 生成目标文件

gcc test.o -o test.out 		# 生成链接后文件 test.out， -o 生成目标文件
```



**`-D` 调试，与宏搭配使用。**

```c
// test.c
# include <stdio.h>

main()
{
    printf("hello world\n");
    #ifdef DEBUG
    printf("DEBUGING...");
    #endif
}

// gcc test.c -o test
// hello world

// gcc test.c -o test -D DEBUG
// hello world
// DEBUGING...

// 等价于
// test.c
# include <stdio.h>
# define DENUG 	// 定义一个宏
main()
{
    printf("hello world\n");
    #ifdef DEBUG 	// 若该宏被定义，则执行这部分代码
    printf("DEBUGING...");
    #endif
}
```

**`-I` 指定头文件目录**

# 静态库和动态库

## 库文件

- 计算机上的一类文件，可以简单地把库文件看成一种代码仓库，它提供给使用者一些可以直接拿来用的变量、函数或类。
- 库是一种特殊的程序，编写上和一般地程序没有较大的区别，但是不能单独运行。
- 库文件有两种“静态库和动态库（共享库）。
  - 静态库：在程序的链接阶段被复制到程序中；
  - 动态库：在程序的链接阶段没有被复制到程序中，在程序的运行阶段（调用动态库 api 时）由系统动态加载到内存中供程序调用，通过 ldd (list dynamic depencencies) 命令检查动态库依赖关系。
- 库的好处：①代码保密。②方便部署和开发。
- 库文件要和头文件（说明库中 api）一起分发。




## 静态库

### 1. 库文件命名规则

- linux：libxxx.a
- windows：libxxx.lib

### 2. 静态库的制作

- gcc 获取 .o 文件

- 将 .o 文件打包（使用 ar 工具，archive）

  ```shell
  ar rsc libxxx.a xxx.o xxx.o 	// ar rsc 库名 .o文件们
  ```

  r - 将文件插入到备存文件中

  c - 建立备存文件

  s - 索引

### 3. 静态库的使用 - 例子

现有四个程序分别实现加减乘除，将它们制作成一个静态库，库名为 `calc`，库文件名为 `libcalc.a`。

<img src="./assets/image-20231016152726842.png" alt="image-20231016152726842" style="zoom: 67%;" />

```shell
gcc -c add.c sub.c mult.c div.c 	// 生成 .o 文件
```

<img src="./assets/image-20231016152803667.png" alt="image-20231016152803667" style="zoom:67%;" />

```shell
ar rcs libcalc.a add.o sub.o mult.o div.o 	// 制作库文件 libcalc.a
```

<img src="./assets/image-20231016153117843.png" alt="image-20231016153117843" style="zoom:67%;" />

在 library 文件夹下尝试使用静态库：

<img src="./assets/image-20231016153210561.png" alt="image-20231016153210561" style="zoom:67%;" />

直接编译链接 main.c 文件，报错找不到头文件，使用 `-I` 参数指定头文件目录

<img src="./assets/image-20231016154021540.png" alt="image-20231016154021540" style="zoom:67%;" />

能够找到头文件，但是找不到定义的函数，使用 `-L` 和 `-l` 参数指定库和库路径。

<img src="./assets/image-20231016154045184.png" alt="image-20231016154045184" style="zoom:67%;" />

成功执行程序。

<img src="./assets/image-20231016154244632.png" alt="image-20231016154244632" style="zoom:67%;" />

## 动态库



### 1. 库文件命名规则

linux：libxxx.so （在 linux 下是一个可执行文件）

windows：libxxx.dll

### 2. 动态库的制作

- gcc 获取 .o 文件（要求是与位置无关的代码）

  ```shell
  gcc -c -fpic/-fPIC a.c b.c 	# -fpic/-fPIC 生成与位置无关的代码
  ```

- gcc 制作动态库

  ```shell
  gcc -shared a.o b.o -o libcalc.so
  ```

### 3. 动态库的使用 - 例子

现有四个程序分别实现加减乘除，将它们制作成一个动态库，库名为 `calc`，库文件名为 `libcalc.so`。

<img src="./assets/image-20231017193958446.png" alt="image-20231017193958446" style="zoom:90%;" />

```
gcc -c -fpic/-fPIC add.c sub.c mult.c div.c 	// 生成 .o 文件
```

<img src="./assets/image-20231017194034877.png" alt="image-20231017194034877" style="zoom:90%;" />

```
gcc shared *.o -o libcalc.so 	// 制作库文件 libcalc.so
```

<img src="./assets/image-20231017194124307.png" alt="image-20231017194124307" style="zoom:90%;" />

尝试在 `main.c` 程序中使用动态库：

<img src="./assets/image-20231017194235458.png" alt="image-20231017194235458" style="zoom:90%;" />

直接编译链接 main.c 文件，报错找不到头文件，使用 `-I` 参数指定头文件目录

![image-20231017194311515](./assets/image-20231017194311515.png)

能够找到头文件，但是找不到定义的函数，使用 `-L` 和 `-l` 参数指定库和库路径。

<img src="./assets/image-20231017194335449.png" alt="image-20231017194335449" style="zoom:90%;" />

成功编译程序，但是运行失败。

<img src="./assets/image-20231017194459951.png" alt="image-20231017194459951" style="zoom:90%;" />

编译时不会报错（因为编译时不连接动态库），但是运行时报错找不到动态库。

![image-20231017194516609](./assets/image-20231017194516609.png)

通过 `ldd` 命令查看可执行程序的依赖关系：

<img src="./assets/image-20231017195233576.png" alt="image-20231017195233576" style="zoom: 90%;" />

为了解决这个问题，要在 **动态载入器** 的搜索范围内增加动态库的绝对地址：

```shell
# 1. expert配置环境变量 （export 配置环境变量仅在当前终端有效）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib
# $LD_LIBRARY_PATH: 表示在这之后添加

# 2. 用户级别修改 （在.bashrc 中的修改在 用户 级别是永久的）
vim .bashrc 	# 末尾增加 "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib"

. .bashrc / source .bashrc

# 3. 系统级别修改 （在 /etc/profile 中的修改在 系统 级别是永久的）
sudo vim /etc/profile 	# 末尾增加 "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib"
sudo source /etc/profile

# 4. 在 ld.so... 中增加
```



![image-20231017200852793](./assets/image-20231017200852793.png)

![image-20231017200902512](./assets/image-20231017200902512.png)



























1.5结束 

