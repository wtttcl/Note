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
  - 静态库：在程序的链接阶段被复制到程序中；（一般比较小）
  - 动态库：在程序的链接阶段没有被复制到程序中，在程序的运行阶段（调用动态库 api 时）由系统动态加载到内存中供程序调用，通过 ldd (list dynamic depencencies) 命令检查动态库依赖关系。（一般比较大）
- 库的好处：①代码保密。②方便部署和开发。
- 库文件要和头文件（说明库中 api）一起分发。



**程序编译成可执行程序的过程：**

<img src="./assets/image-20231018201840024.png" alt="image-20231018201840024" style="zoom:80%;" />



## 静态库

<img src="./assets/image-20231018202003786.png" alt="image-20231018202003786" style="zoom:80%;" />

- 静态库在链接阶段会被打包复制到可执行程序中。

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

<img src="./assets/image-20231018202108300.png" alt="image-20231018202108300" style="zoom:80%;" />

- gcc 在进行链接的时候，动态库的代码不会被打包到可执行程序中，而是在程序启动后，当调用了动态库中的 API 时，动态库会被动态加载到内存中。
- 当系统加载动态库时，不仅需要知道库的名字，还需要知道库的绝对路径。
- 系统的 **动态载入器** 可以获取动态库的绝对路径。对于 elf 格式的可执行程序来说，`ld-linux.so` 承担动态载入器的角色。它先后搜索 elf 文件的 **`DT_RPATH` 段**（一般不访问）、**环境变量 `LD_LIBRARY_PATH`**、**`/etc/ld.so.cache` 文件列表**、**`/lib/` 和 `/usr/lib/` 目录** 来寻找库文件，并将其载入内存。
- elf 文件是一种用于二进制文件、可执行文件、目标代码、共享库和核心转储格式文件的文件格式。

- 通过 `ldd` (list dynamic dependencies) 命令可以检查动态库依赖关系。

### 1. 库文件命名规则

linux：libxxx.so （在 linux 下是一个可执行文件）

windows：libxxx.dll

### 2. 动态库的制作

- gcc 获取 .o 文件（要求是 **与位置无关** 的代码）

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

为了解决这个问题，要在 **动态载入器** 的搜索范围内 **增加动态库的绝对地址**：

```shell
# 1. 在 LD_LIBRARY_PATH 中添加库文件的绝对路径
## a. expert 配置环境变量 （export 配置环境变量仅在当前终端有效）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib
# $LD_LIBRARY_PATH: 表示在这之后添加新路径

## b. 用户级别修改 （在.bashrc 中的修改在 用户 级别是永久的）
vim .bashrc 	# 末尾增加 "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib"
. .bashrc / source .bashrc

## c. 系统级别修改 （在 /etc/profile 中的修改在 系统 级别是永久的）
sudo vim /etc/profile 	# 末尾增加 "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib"
sudo source /etc/profile


# 2. 在 /etc/ld.so.cache 中增加（不能直接访问，在 /etc/ld.so.conf.d 中增加）
sudo vim /etc/ld.so.conf.d 	# 直接在末尾增加路径 "/home/ubuntu/lyy/cpp/webserver/lesson06/library/lib"
sudo ldconfig


# 3. 直接把动态库文件放到 /lib/ huo /usr/lib/ 目录下 （不建议，这两个目录下存在很多系统自带的库文件，防止重名覆盖）
```

在增加动态库文件的绝对路径后，可以看到该库文件被正常加载到内存中了。

![image-20231017200852793](./assets/image-20231017200852793.png)

app 可执行程序可以正常执行了。

![image-20231017200902512](./assets/image-20231017200902512.png)

## 静态库和动态库的对比

### 1. 静态库的优缺点

**优点：**

- 静态库链接阶段被打包到可执行程序中，所以程序运行时加载快。
- 发布程序不需要提供静态库，移植方便。

**缺点：**

- 消耗系统资源，浪费内存。（可能多处复制）
- 更新、部署、发布麻烦。（需要重新编译）

<img src="./assets/image-20231018202605463.png" alt="image-20231018202605463" style="zoom:80%;" />

### 2. 动态库的优缺点

**优点：**

- 可以实现进程间资源共享（动态库时共享库）
- 更新、部署、发布简单。（不需要重新编译）
- 可以控制何时加载动态库。

**缺点：**

- 加载速度较慢。
- 发布程序时需要提供依赖的动态库。（可执行程序执行时需要动态库，而静态库已经包含在程序中了）

<img src="./assets/image-20231018202901850.png" alt="image-20231018202901850" style="zoom:80%;" />

# Makefile

- Makefile 文件定义了一系列的规则来指定哪些文件需要先编译，哪些文件需要后编 译，哪些文件需要重新编译，甚至于进行更复杂的功能操作，因为 Makefile 文件就 像一个 Shell 脚本一样，也可以执行操作系统的命令。
- 自动化编译。一旦写好，只需要一个 make 命令，整 个工程完全自动编译，极大的提高了软件开发的效率。
- make 是一个命令工具，是一个 解释 Makefile 文件中指令的命令工具。

## 简单的 Makfile

- 文件命名：makefile 或 Makefile

- 规则：一个 Makefile 文件中可以有一个或多个规则。但是 **Makefile 中的其他规则一般都是为第一条规则服务的**。

  <div align=left><img src="./assets/image-20231018205232440.png" alt="image-20231018205232440" /></div>

  - 目标：最终要生成的文件（伪目标除外）
  - 依赖：生成目标所需要的文件或目标
  - 命令：通过执行命令对依赖操作生成目标（必须有 Tab 缩进）

### 1. 例子

现有四个程序分别实现加减乘除和一个使用加减乘除的程序 `main.c`，现在编写一个makefile，用 make 实现编译。

![image-20231018210027984](./assets/image-20231018210027984.png)

编写 Makefile 文件

![image-20231018210048275](./assets/image-20231018210048275.png)

执行 make

![image-20231018210038671](./assets/image-20231018210038671.png)

## Makefile 工作原理

<img src="./assets/image-20231018205232440.png" alt="image-20231018205232440" />

- 命令在执行之前，首先会检查规则中的依赖是否存在。如果不存在，则向下检查其他规则；如果存在，则指直接执行命令。==**与第一条规则没有任何关系的规则不会执行！**==
- 检查更新。在执行规则的命令时，会比较目标和依赖文件的时间。如果依赖的时间比目标的时间晚，则会重新执行命令生成目标；如果依赖的时间比目标的时间早，则不进行更新。

Makefile：

![image-20231018210857665](./assets/image-20231018210857665.png)

命令执行：

![image-20231018210843896](./assets/image-20231018210843896.png)

检查更新：（这种写法可以优于例子中的写法，因为检查更新的存在）

![image-20231018211151923](./assets/image-20231018211151923.png)

## Makefile 的变量

### 1. 自定义变量

- ```makefile
  变量名=变量值 	# var=hello
  ```

  

### 2. 预定义变量

- AR：归档维护程序的名称，默认值为 ar；
- CC：C 编译器的名称，默认值为 cc；
- CXX：C++ 编译器的名称，默认值为 g++；
- $@：目标的完整名称；
- $<：第一个依赖文件的名称（不带后缀）；
- $^：所有的依赖文件；

### 3. 获取变量的值

- ```makefile
  $(变量名) 	# $(var)
  ```

## Makefile 的模式匹配

```makefile
%.o:%.c
```

- %：通配符，匹配一个字符串；
- 两个 % 匹配的是同一个字符串；



# Makefile 的函数

$(wildcard PATTERN...)

## 简化的 Makefile

![image-20231018210857665](./assets/image-20231018210857665.png)

用变量简化后



