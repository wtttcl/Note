# gcc

GCC 原名 GNU C 语言编译器（GNU C Compiler），只能处理 C 语言。但其很快扩展，变得可处理 C++，后来又扩展为能够支持更多编程语言，如 Pascal、Objective -C、Java、Go 以及各类处理器架构上的汇编语言等，所以改名 GNU 编译器套件（GNU Compiler Collection）。



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

<img src="./assets/03cb0d48c26a96a88c4d6141f65ae4a.jpg" alt="03cb0d48c26a96a88c4d6141f65ae4a" style="zoom: 35%;" />

- 预处理后源代码：删去注释、宏替换等
- **`-o` 本质上是一个重命名选项，直接执行 `gcc test.c` 也会生成可执行文件 `a.out`**。
- **当只有一个目标代码时，`-o`可有可无。**

```shell
gcc test.c -E -o test.i 	# 生成预处理后文件 test.i， -E 预处理 -o 生成目标文件

gcc test.i -S -o test.s 	# 生成编译后文件 test.s， -S 编译 -o 生成目标文件

gcc test.s -c -o test.o 	# 生成汇编后文件 test.o， -S 编译 -o 生成目标文件

gcc test.o -o test.out 		# 生成链接后文件 test.out， -o 生成目标文件


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

- **`-I` 指定头文件目录**
- **`-L` 指定库路径**
- **`-l` 指定库**

- **`-D` 调试，与宏搭配使用。**

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

<hr style="border:3px #6CA6CD double;">



# 静态库和动态库

## 库文件

- 计算机上的一类文件，可以简单地把库文件看成一种代码仓库，它提供给使用者一些可以直接拿来用的变量、函数或类。
- 库是一种特殊的程序，编写上和一般地程序没有较大的区别，但是不能单独运行。
- 库文件有两种“静态库和动态库（共享库）。
  - 静态库：在程序的  **链接阶段** 被 **复制** 到程序中；（一般比较小）
  - 动态库：在程序的链接阶段没有被复制到程序中，在程序的 **运行阶段**（调用动态库 api 时）由系统 **动态加载到内存中供程序调用**，通过 ldd (list dynamic depencencies) 命令检查动态库依赖关系。（一般比较大）
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

<hr style="border:3px #6CA6CD double;">

# Makefile

- Makefile 文件定义了一系列的规则来指定哪些文件需要先编译，哪些文件需要后编译，哪些文件需要重新编译，甚至于进行更复杂的功能操作，因为 Makefile 文件就 像一个 Shell 脚本一样，也可以执行操作系统的命令。
- 自动化编译。一旦写好，只需要一个 make 命令，整 个工程完全自动编译，极大的提高了软件开发的效率。
- make 是一个命令工具，是一个 解释 Makefile 文件中指令的命令工具。

## 简单的 Makfile

- 文件命名：makefile 或 Makefile

- 规则：一个 Makefile 文件中可以有一个或多个规则。但是 **Makefile 中的其他规则一般都是为第一条规则服务的**。

  <div align=left><img src="./assets/image-20231018205232440.png" alt="image-20231018205232440" /></div>

  - 目标：最终要生成的 **文件**（伪目标除外）
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

- clean 规则：将 clean 文件设置成 **伪文件**，避免与其他名为 clean 的文件冲突。clean 规则不需要依赖。

  ![image-20231019100145277](./assets/image-20231019100145277.png)

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
- 一个规则中的两个或多个 % 匹配的是同一个字符串；



# Makefile 的函数

**1. `wildcard`**

```makefile
$(wildcard PATTERN...) 	# (函数名 参数)
```

- 获取指定目录下指定类型的文件列表，返回一个文件列表（空格间隔）

- 参数 PATTERN 指的是某个或多个目录下的对应的某种类型的文件，如果有多个目录，一般使用空格间隔

- e.g.

  <img src="./assets/image-20231019092731403.png" alt="image-20231019092731403" style="zoom:80%;" />



**2. `patsubst`**

```makefile
$(patsubst <pattern>,<replacement>,<text>)
```

- 查找 `<text>` 中的单词（以 空格/Tab/回车/换行 分隔）是否符合模式 `<pattern>`，如果符合，则用 `<replacement>` 替换，返回被替换后的字符串；

- `<pattern>` 可以包括通配符 '%' 表示任意长度的字符串（'\%' 代表 '%'），若 `<replacement>` 也含有 '%'，那么它们表示的是同一个字符串；

- e.g.

  <img src="./assets/image-20231019093458669.png" alt="image-20231019093458669" style="zoom:80%;" />

## 简化的 Makefile

简化前的 Makefile（两种写法）：

<img src="./assets/image-20231018210048275.png" alt="image-20231018210048275" style="zoom:90%;" />

<img src="./assets/image-20231018210857665.png" alt="image-20231018210857665" style="zoom:90%;" />

用变量简化后：

<img src="./assets/image-20231019091939040.png" alt="image-20231019091939040" style="zoom:90%;" />

<img src="./assets/image-20231019091819295.png" alt="image-20231019091819295" style="zoom:90%;" />

编译执行：

<img src="./assets/image-20231019091702838.png" alt="image-20231019091702838" style="zoom:90%;" />

<img src="./assets/image-20231019091730739.png" alt="image-20231019091730739" style="zoom:90%;" />

用函数简化 Makefile 后：

![image-20231019094213441](./assets/image-20231019094213441.png)

![image-20231019094154640](./assets/image-20231019094154640.png)

增加 clean 规则删除中间生成的 .o 文件。

![image-20231019094709307](./assets/image-20231019094709307.png)

![image-20231019094353584](./assets/image-20231019094353584.png)

<hr style="border:3px #6CA6CD double;">

# GDB

- GDB 是由 GNU 软件系统社区提供的调试工具，同 GCC 配套组成了一套完整的开发环 境，GDB 是 Linux 和许多类 Unix 系统中的标准开发环境。

- 一般来说，GDB 主要帮助你完成下面四个方面的功能： 

  - 启动程序，可以按照自定义的要求随心所欲的运行程序
  - 可让被调试的程序在所指定的调置的断点处停住（断点可以是条件表达式） 
  - 当程序被停住时，可以检查此时程序中所发生的事
  - 可以改变程序，将一个 BUG 产生的影响修正从而测试其他 BUG

- 通常，在为调试而编译时，我们会（）关掉编译器的优化选项（`-O`）， 并打开调 试选项（`-g`）。另外，`-Wall`在尽量不影响程序行为的情况下选项打开所有 warning，也可以发现许多问题，避免一些不必要的 BUG。

  ```makefile
  gcc -g -Wall program.c -o program 	# `-g` 选项的作用是在可执行文件中加入源代码的信息，比如可执行文件中第几条机器指令对应源代码的第几行，但并不是把整个源文件嵌入到可执行文件中，所以在调试时必须保证 gdb 能找到源文件。
  ```



## GDB 常用命令

### 1. 启动和退出

```shell
gdb 可执行程序名 	# gdb main
quit
```

### 2. 给程序设置命令行参数 / 获取设置的命令行参数

```shell
set args 10 20
show args
```

### 3. gdb 使用帮助

```shell
help
```

### 4. 查看当前文件代码

```shell
# 查看当前文件代码
list/l （从默认位置显示）
list/l 行号 （从指定的行显示）
list/l 函数名（从指定的函数显示）

# 查看非当前文件代码
list/l 文件名:行号
list/l 文件名:函数名
```

### 5. 设置显示的行数

```shell
show list/listsize
set list/listsize 行数
```

### 6. 断点

```shell
# 设置断点
b/break 行号
b/break 函数名
b/break 文件名:行号
b/break 文件名:函数

# 查看断点
i/info b/break

# 删除断点
d/del/delete 断点编号

# 设置断点无效
dis/disable 断点编号

# 设置断点生效
ena/enable 断点编号

# 设置条件断点（一般用在循环的位置）
b/break 10 if i==5
```

### 7. 运行

```shell
# 运行GDB程序
start（程序停在第一行）
run（遇到断点才停）

# 继续运行，到下一个断点停
c/continue

# 向下执行一行代码（不会进入函数体）
n/next

# 向下单步调试（遇到函数进入函数体）
s/step
finish（跳出函数体）
```

### 8. 查看变量

```shell
# 变量操作
p/print 变量名（打印变量值）
ptype 变量名（打印变量类型）

# 其它操作
set var 变量名=变量值 （循环中用的较多）
until （跳出循环）
```

## GDB 多进程调试

- GDB 默认跟踪父进程，子进程代码直接执行。
- 设置调试父进程或子进程： `set follow-fork-mode [child | parent]`
- 设置调试模式：`set detach-on-fork [on | off]` 。默认为 on，表示调试当前进程的时候，其它的进程继续运行，如果为 off，调试当前进程的时候，其它进程被 GDB 挂起。
- 查看调试的进程：`info inferiors`（当前调试进程含有 *）
- 切换当前调试的进程：`inferior id`
- 使进程脱离 GDB 调试：`detach inferiors id`

<hr style="border:3px #6CA6CD double;">

# 文件 IO

## 1. 标准 C 库 IO 函数 与 linux 系统 IO 的关系

标准 C 库 IO 函数通过 FILE 文件指针进行文件操作。

<img src="./assets/image-20231023202203474.png" alt="image-20231023202203474" style="zoom:80%;" />

<img src="./assets/image-20231023202409922.png" alt="image-20231023202409922" style="zoom:80%;" />

---



## 2. c 程序的虚拟存储地址空间布局

C 程序的组成：

- **正文段**。CPU 执行的机器指令部分。通常是可共享且只读的。
- **初始化数据段**，又称数据段。包含程序中需明确赋初值的变量，如，**全局变量**。
- **未初始化数据段**，称 bss 段。包含 **函数外的未初始化的变量**，如，`long sum[1000];`。在程序开始执行之前，内核将此段中的数据初始化为 0 或空指针。
- **栈**。包含自动变量（非静态或外部变量）以及每次函数调用时需要保存的信息（返回地址、调用者的环境信息）。最近被调用的函数会在栈上为其自动变量和临时变量分配存储空间。
- **堆**。动态存储分配在堆中进行。

<img src="./assets/image-20231105143509973.png" alt="image-20231105143509973" style="zoom:90%;" />

- 只有正文段和初始化数据段需要从磁盘程序文件中读入。未初始化的数据段不需要保存在磁盘程序文件中，因为他们会被内核初始化为 0 或空指针。



*若定义一个局部变量为自动变量,这意味着每次执行到定义该变量都回产生一个新变量,并对他重新初始化*

**以 32 位计算机为例：**

32 位计算机会为每个进程分配 4G 的虚拟地址空间，包括内核区（只能通过 **系统调用** 进行操作）和用户区。虚拟地址空间最终会被 MMU 映射到物理地址空间。

<img src="./assets/image-20231023203022758.png" alt="image-20231023203022758" style="zoom: 70%;" />

---



## 3. 文件描述符

![image-20231023204302508](./assets/image-20231023204302508.png)

- **文件描述符表** 被保存在进程的内核区，由内核区内的 **PCB 进程控制块** 维护。
- 文件描述符表是一个数组，大小默认是 1024（每个进程默认最多可以同时打开 1024 个文件）。
- 文件描述符表中：0（标准输入）、1（标准输出）、2（标准错误），默认打开，指向当前终端。
- **一个文件可以被同时打开 n 次，每次打开得到的文件描述符是不一样的。**

---



## 4. `errno`

`errno`：属于 linux 系统函数库，是一个全局变量，**记录最近的错误号**。可以调用 `perror` 函数获取错误号对应的错误描述。

```c
/*
#include <stdio.h>

void perror(const char *s);     // 打印 errno 对应的错误描述，没有返回值。
  参数：
    - 用户描述（最后打印为 s:错误描述）
    
  作用：
    - 输出 “用户描述：错误描述”。
*/
```

---



## 5. `open` 

### a. 查看 `open` 函数

```shell
man 2 open
```



### b. 头文件

```c
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
```



### c. `open`

有两个 `open` 函数，一个用来打开已经存在的文件，一个用来创建新的文件。

```c
int open(const char *pathname, int flags);
/*
  参数：
    - pathname：文件路径
    - flags：对文件操作的权限设置及其他设置
      - 必选项（互斥，必须选一个）：O_RDONLY 只读, O_WRONLY 只写, or O_RDWR 读写；
      - 可选项：O_APPEND 追加，...
    
  返回值：
    - 调用成功，返回一个新的文件描述符；调用失败，返回 -1
  
  作用：
    - 打开文件，返回文件描述符
*/
```



```c
int open(const char *pathname, int flags, mode_t mode);
/*
  参数：
    - pathname：要创建的文件的路径
    - flags：对文件操作的权限设置及其他设置
      - 必选项（互斥，必须选一个）：O_RDONLY 只读, O_WRONLY 只写, or O_RDWR 读写；
      - 可选项：O_APPEND 追加，O_CREAT 文件不存在则创建
    - mode：八进制数，表示用户对创建的新文件的操作权限，最终的权限为 mode & ~umask。e.g. 0777（最高权限）
    
  返回值：
    - 调用成功，返回一个新的文件描述符；调用失败，返回 -1
  
  作用：
    - 创建并打开文件，返回文件描述符
*/
```



e.g.

```c
/*
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// 打开一个已经存在的文件
int open(const char *pathname, int flags);
  参数：
    - pathname：文件路径
    - flags：对文件操作的权限设置及其他设置
      - 必选项（互斥，必须选一个）：O_RDONLY 只读, O_WRONLY 只写, or O_RDWR 读写；
      - 可选项：O_APPEND 追加，...
    
  返回值：
    - 调用成功，返回一个新的文件描述符；调用失败，返回 -1
  
  作用：
    - 打开文件，返回文件描述符
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h> 	// c 标准输入输出头文件
#include <unistd.h> 	// unix 标准库头文件，包含一些常用的系统调用函数和符号常量，如 write、read、fork、exec 等，以及对文件描述符、进程控制和文件操作的定义。

int main()
{
    int fd = open("a.txt", O_RDONLY);

    if(fd == -1)
    {
        perror("open");     // print error desc
    } 

    // close file desc
    close(fd);
}
```

运行结果：

<img src="./assets/image-20231023211359839.png" alt="image-20231023211359839" style="zoom:80%;" />

```c
/*
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// 创建一个新文件，并打开
int open(const char *pathname, int flags, mode_t mode);
  参数：
    - pathname：要创建的文件的路径
    - flags：对文件操作的权限设置及其他设置
      - 必选项（互斥，必须选一个）：O_RDONLY 只读, O_WRONLY 只写, or O_RDWR 读写；
      - 可选项：O_APPEND 追加，O_CREAT 文件不存在则创建
    - mode：八进制数，表示用户对创建的新文件的操作权限，最终的权限为 mode & ~umask。e.g. 0777（最高权限）
    
  返回值：
    - 调用成功，返回一个新的文件描述符；调用失败，返回 -1
  
  作用：
    - 创建并打开文件，返回文件描述符
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
    int fd = open("a.txt", O_RDWR | O_CREAT, 0777);

    if(fd == -1)
    {
        perror("open");     // print error desc
    } 

    // close file desc
    close(fd);
}
```

---



## 6. `close`

### a. 查看 `close` 函数

```shell
man 2 close
```



### b. 头文件

```c
#include <unistd.h>
```



### c. `close`

```c
int close(int fd);  //关闭文件，并使得文件描述符可以被再次使用
/*
  参数：
    - fd：要关闭的文件描述符
    
  作用：
    - 关闭文件，释放文件描述符
*/

```

---



## 7. `read` 和 `write`

### a. 查看  `read` 和 `write` 函数

```shell
man 2 read 	# man 2 write
```



### b. 头文件

```c
#include <unistd.h>
```



### c. `read`

```c
ssize_t read(int fd, void *buf, size_t count);
/*  
  参数：
    - fd：文件描述符
    - buf：缓冲区
    - count：指定的 buf 数组的大小
    
  返回值：
    - 调用成功，返回读取的字节数（字节数为 0, 表示文件读取完毕）；调用失败，返回 -1，并且设置 errno。
    
  作用：
    - 从文件中读取给定数量的字节。
*/
```

### d. `write`

```c
ssize_t write(int fd, const void *buf, size_t count);
/*
  参数：
    - fd：文件描述符
    - buf：缓冲区
    - count：要写入的数据大小
    
  返回值：
    - 调用成功，返回写入的字节数（字节数为 0, 表示文件读取完毕）；调用失败，返回 -1，并且设置 errno。
    
  作用：
    - 向文件中写入给定数量的字节。
*/
```



```c
/**
#include <unistd.h>

ssize_t read(int fd, void *buf, size_t count);
  参数：
    - fd：文件描述符
    - buf：缓冲区
    - count：指定的 buf 数组的大小
    
  返回值：
    - 调用成功，返回读取的字节数（字节数为 0, 表示文件读取完毕）；调用失败，返回 -1，并且设置 errno。
    
  作用：
    - 从文件中读取给定数量的字节。
*/

/*
#include <unistd.h>

ssize_t write(int fd, const void *buf, size_t count);
  参数：
    - fd：文件描述符
    - buf：缓冲区
    - count：要写入的数据大小
    
  返回值：
    - 调用成功，返回写入的字节数（字节数为 0, 表示文件读取完毕）；调用失败，返回 -1，并且设置 errno。
    
  作用：
    - 向文件中写入给定数量的字节。
*/

#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main()
{
    // 打开要读取的文件
    int srcfd = open("english.txt", O_RDONLY);
    if(srcfd == -1)
    {
        perror("open");
        return -1;
    }

    // 创建并打开要写入的文件
    int desfd = open("cpy.txt", O_RDWR |O_CREAT, 0777);
    if(desfd == -1)
    {
        perror("open");
        return -1;
    }

    // 创建缓冲区
    char buf[1024] = {0};

    int len = 0;
    while((len=read(srcfd, buf, sizeof(buf))) > 0)  // 只要还没读完，就一直写入
    {
        write(desfd, buf, len);     // 读入多少 len 写入多少 len
    }

    // 关闭文件
    close(srcfd);
    close(desfd);
    return 0;
}
```

![image-20231024103828530](./assets/image-20231024103828530.png)

---



## 8. `lseek`

### a. 查看 `lseek` 函数

```shell
man 2 lseek
```



### b. 头文件

```c
#include <sys/types.h>
#include <unistd.h>
```



### c. `lseek`

```c
off_t lseek(int fd, off_t offset, int whence);
/*
  参数：
    fd：文件描述符，通过 open 得到;
    offset：偏移量
    whence：标记，设置文件指针偏移量
      - SEEK_SET：设置偏移量：offset;
      - SEEK_CUR：设置偏移量：当前位置 + offset;
      - SEEK_END：设置偏移量：文件大小 + offset;

  返回值：
    - 调用成功，返回文件指针最终的位置；调用失败，返回-1，并且设置errno。

  作用：
    - 移动文件指针到文件头：lseek(fd, 0, SEEK_SET);
    - 获取当前文件指针的位置：lseek(fd, 0, SEEK_CUR);
    - 获取文件长度：lseek(fd, 0, SEEK_END);
    - 拓展文件长度：lseek(fd, 100, SEEK_END);   // 最好在末尾写入一些字符才能实现拓展
*/
```

**举个例子：**

```c
/*
#include <sys/types.h>
#include <unistd.h>

off_t lseek(int fd, off_t offset, int whence);
  参数：
    fd：文件描述符，通过 open 得到;
    offset：偏移量
    whence：标记，设置文件指针偏移量
      - SEEK_SET：设置偏移量：offset;
      - SEEK_CUR：设置偏移量：当前位置 + offset;
      - SEEK_END：设置偏移量：文件大小 + offset;

  返回值：
    - 调用成功，返回文件指针最终的位置；调用失败，返回-1，并且设置errno。

  作用：
    - 移动文件指针到文件头：lseek(fd, 0, SEEK_SET);
    - 获取当前文件指针的位置：lseek(fd, 0, SEEK_CUR);
    - 获取文件长度：lseek(fd, 0, SEEK_END);
    - 拓展文件长度：lseek(fd, 100, SEEK_END);   // 最好在末尾写入一些字符才能实现拓展
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h> 	// 文件操作函数 open
#include <stdio.h> 	// perror

int main()
{
    int fd = open("hello.txt", O_RDWR);
    if(fd == -1)
    {
        perror("open");
        return -1;
    }

    int ret = lseek(fd, 100, SEEK_END);
    if(ret == -1)
    {
        perror("lseek");
        return -1;
    }

    // 写入数据
    write(fd, " ", 1);
    
    // 关闭文件
    close(fd);
    return 0;
}
```

若没有在拓展后的文件末尾写入一些字符，文件并没有被成功拓展：

<img src="./assets/image-20231031134757004.png" alt="image-20231031134757004" style="zoom:85%;" />

调用 `write` 函数在文件末尾写入字符后，可以看到文件长度发生了变化，文件长度变为：**原来长度 + 拓展的长度 + 写入的字符串的长度。**

<img src="./assets/image-20231031134808429.png" alt="image-20231031134808429" style="zoom:85%;" />

---



## 9. `stat` 和 `fstat` 和 `lstat`

### a. 查看 `stat` 和 `fstat` 和 `lstat`函数

```shell
man 2 stat/fstat/lstat
```



### b. 头文件

```c
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
```



### c. `stat`

```c
int stat(const char *pathname, struct stat *statbuf);
/*
  参数：
    - pathname：文件路径
    - statbuf：结构体变量，接收获取的文件相关信息

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置 errno

  作用：
    - 获取一个文件的相关信息（当文件是一个软连接时，获取的是软链接指向的文件的信息）
*/
```

### d. `fstat`

```c
int fstat(int fd, struct stat *statbuf);
/*
  参数：
    - fd：文件描述符，open 获得
    - statbuf：结构体变量，接收获取的文件相关信息

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置 errno
  
  作用：
    - 获取一个文件的相关信息（除参数外，作用于 stat 相同）
*/
```

### e. `lstat`

```c
int lstat(const char *pathname, struct stat *statbuf);
/*
  参数：
    - pathname：文件路径
    - statbuf：结构体变量，接收获取的文件相关信息

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置 errno
  
  作用：
    - 获取一个文件的相关信息（当文件是一个软连接时，获取的是软链接本身的信息）
*/
```

### f. `struct stat`：

```c
// 就是命令 stat filename 输出的信息
struct stat {
               dev_t     st_dev;         /* 文件的设备编号 */
               ino_t     st_ino;         /* 节点号 */
               mode_t    st_mode;        /* 文件的类型和存取的权限 */
               nlink_t   st_nlink;       /* 链接到该文件的硬链接的数据 */
               uid_t     st_uid;         /* 用户ID */
               gid_t     st_gid;         /* 组ID */
               dev_t     st_rdev;        /* 设备文件的设备编号 */
               off_t     st_size;        /* 文件大小（字节数） */
               blksize_t st_blksize;     /* I/O文件系统的块大小 */
               blkcnt_t  st_blocks;      /* 分配的块的数量（512B） */

               /* Since Linux 2.6, the kernel supports nanosecond
                  precision for the following timestamp fields.
                  For the details before Linux 2.6, see NOTES. */

               struct timespec st_atim;  /* 最后一次访问时间 */
               struct timespec st_mtim;  /* 最后一次修改时间 */
               struct timespec st_ctim;  /* 最后一次改变属性时间 */

           #define st_atime st_atim.tv_sec      /* Backward compatibility */
           #define st_mtime st_mtim.tv_sec
           #define st_ctime st_ctim.tv_sec
           };

```

### g. `st_mode`：

`st_mode` 是一个 $16$ 位的二进制串，用来表示文件类型（7种，前四位和 `S_IFMT` 掩码位与获取）和文件权限（12位）。

<img src="./assets/image-20231031140703954.png" alt="image-20231031140703954" style="zoom: 70%;" />

### h. 举个例子

```c
/*
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>

int stat(const char *pathname, struct stat *statbuf);
  参数：
    - pathname：文件路径
    - statbuf：结构体变量，接收获取的文件相关信息

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置 errno
    
  作用：
    - 获取一个文件的相关信息

  int lstat(const char *pathname, struct stat *statbuf);
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

int main()
{
    struct stat statbuf;
    int ret = stat("hello.txt", &statbuf);

    if(ret == -1)
    {
        perror("stat");
        return -1;
    }

    printf("size:%d\n", statbuf.st_size);

    return 0;
}
```

运行结果如下：

![image-20231031141240075](./assets/image-20231031141240075.png)

---



## 10. 模拟实现 `ls -l` 命令

```c
/*
模拟实现 ls -l 指令

-rwxrwxr-x 1 ubuntu ubuntu    113 十月   31 13:47 a.txt
*/

#include <sys/types.h>
#include <sys/stat.h>   //
#include <stdio.h>  //
#include <unistd.h>     //
#include <pwd.h>    // getpwuid
#include <grp.h>    // getgrgid
#include <time.h>
#include <string.h>

int main(int argc, char * argv[])
{
    // 参数输入是否正确
    if(argc < 2)
    {
        printf("usage: %s <filename>\n", argv[0]);
        return -1;
    }

    // 调用 stat 函数获取文件信息
    struct stat st;
    int ret = stat(argv[1], &st);
    if(ret == -1)
    {
        perror("stat");
        return -1;
    }

    // 获取文件类型 (1位) 和文件权限 （9位）
    char perms[11] = {0};   // 最后一位表示字符串结束
    
    switch(st.st_mode & __S_IFMT)
    {
        case __S_IFLNK:     // 符号链接（软链接）
            perms[0] = 'l';
            break;
        case __S_IFDIR:     // 目录
            perms[0] = 'd';
            break;
        case __S_IFREG:     // 普通文件
            perms[0] = '-';
            break;
        case __S_IFBLK:     // 块设备
            perms[0] = 'b';
            break;
        case __S_IFCHR:     // 字符设备
            perms[0] = 'c';
            break;
        case __S_IFSOCK:    // 套接字
            perms[0] = 's';
            break;
        case __S_IFIFO:     // 管道
            perms[0] = 'p';
            break;
        default:
            perms[0] = '?';
            break;
    }
    // 判断文件的访问权限

    // 文件所有者
    perms[1] = (st.st_mode & S_IRUSR) ? 'r' : '-';
    perms[2] = (st.st_mode & S_IWUSR) ? 'w' : '-';
    perms[3] = (st.st_mode & S_IXUSR) ? 'x' : '-';

    // 文件所在组
    perms[4] = (st.st_mode & S_IRGRP) ? 'r' : '-';
    perms[5] = (st.st_mode & S_IWGRP) ? 'w' : '-';
    perms[6] = (st.st_mode & S_IXGRP) ? 'x' : '-';
    
    // 其他人
    perms[7] = (st.st_mode & S_IROTH) ? 'r' : '-';
    perms[8] = (st.st_mode & S_IWOTH) ? 'w' : '-';
    perms[9] = (st.st_mode & S_IXOTH) ? 'x' : '-';
    
    // 硬链接数
    int linkNum = st.st_nlink;

    // 文件所有者
    char * fileUser = getpwuid(st.st_uid)->pw_name;

    // 文件所在组
    char * fileGrp = getgrgid(st.st_gid)->gr_name;

    // 文件大小
    long int fileSize = st.st_size;

    // 获取修改的时间
    char * time = ctime(&st.st_mtime);  // 带有换行符
    char mtime[512] = {0};
    strncpy(mtime, time, strlen(time) - 1);     // 去掉换行符

    char * buf[1024];
    sprintf(buf, "%s %d %s %s %ld %s %s", perms, linkNum, fileUser, fileGrp, fileSize, mtime, argv[1]);

    printf("%s\n", buf);

    return 0;
}
```

---



## 11. 文件属性操作函数

### a. `access`

**头文件：**

```c
 #include <unistd.h>
```



**access：**

```c
int access(const char *pathname, int mode);
/*
  参数：
    - pathname：要判断的文件路径
    - mode：要判断的权限
        - R_OK: 判断是否有读权限
        - W_OK: 判断是否有写权限
        - X_OK: 判断是否有执行权限
        - F_OK: 判断文件是否存在
  作用：
    - 判断用户对某个文件是否有某个权限或某个文件是否存在

  返回值：
    - 调用成功，返回0；调用失败，返回-1
*/
```



**e.g.**

```c
/*
#include <unistd.h>

int access(const char *pathname, int mode);
  参数：
    - pathname：要判断的文件路径
    - mode：要判断的权限
        - R_OK: 判断是否有读权限
        - W_OK: 判断是否有写权限
        - X_OK: 判断是否有执行权限
        - F_OK: 判断文件是否存在
  作用：
    - 判断用户对某个文件是否有某个权限或文件是否存在

  返回值：
    - 调用成功，返回0；调用失败，返回-1
*/

#include <unistd.h>
#include <stdio.h> 	// 标准输入输出库的头文件，perror 头文件

int main()
{

    int ret = access("a.txt", F_OK);
    if(ret == -1)
    {
        perror("access");
        return -1;
    }

    printf("success!\n");

    return 0;
}
```



### b. `chmod`

**头文件：**

```c
#include <sys/stat.h>
```



**chmod：**

```c
int chmod(const char *pathname, mode_t mode);
/*
  参数：
    - pathname：需要修改权限的文件路径
    - mode：需要修改的权限值

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 修改某个文件的权限
*/
```



**e.g.**

```c
/*
#include <sys/stat.h>

int chmod(const char *pathname, mode_t mode);
  参数：
    - pathname：需要修改权限的文件路径
    - mode：需要修改的权限值

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 修改某个文件的权限
*/

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>

int main()
{
    int ret = chmod("a.txt", 0775);
    if(ret == -1)
    {
        perror("chmod");
        return -1;
    }

    printf("success!\n");

    return 0;
}
```



### c. `chown`

**头文件：**

```c
#include <unistd.h>
```



**chown：**

```c
int chown(const char *pathname, uid_t owner, gid_t group);
/*
  参数：
    - pathname：需要修改所有者的文件路径
    - owner：新的所有者ID 	（通过 vim /etc/passwd 查询所有用户ID）
    - groud：新的组ID 	（通过 vim /etc/group 查询所有组ID）

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 修改某个文件的所有者
*/
```

### d. `truncate`

**头文件：**

```c
#include <unistd.h>
#include <sys/types.h>
```



**truncate：**

```c
int truncate(const char *path, off_t length);
/*
  参数：
    - path：需要修改的文件路径
    - length：文件最终的大小

  返回值：
    - 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 缩减（会截断）或拓展（空字符填充）文件的尺寸至指定的大小
*/
```

---



## 12. 目录操作函数

### a. `mkdir`

**头文件：**

```c
#include <sys/stat.h>
#include <sys/types.h>
```



```c
int mkdir(const char *pathname, mode_t mode);
/*
  参数：
	- pathname: 创建的目录的路径
	- mode: 目录权限，八进制的数
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno
	
  作用：
    - 创建一个目录
*/
```

**e.g.**

```c
/*
#include <sys/stat.h>
#include <sys/types.h>

int mkdir(const char *pathname, mode_t mode);
  参数：
	- pathname: 创建的目录的路径
	- mode: 目录权限，八进制的数
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno
	
  作用：
    - 创建一个目录
*/

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>

int main() {

    int ret = mkdir("aaa", 0777);

    if(ret == -1) {
        perror("mkdir");
        return -1;
    }

    return 0;
}
```



### b. `rmdir`

**头文件：**

```c
#include <unistd.h>
```



```c
int rmdir(const char *pathname);
/*
  参数：
	- pathname: 要删除的目录路径
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno
	
  作用：
    - 删除空目录（只能删除空目录）
*/
```

### c. `rename`

**头文件：**

```c
#include <stdio.h>
```



```c
int rename(const char *oldpath, const char *newpath);
/*
  参数：
	- oldpath: 要重命名的目录路径
    - newpath：重命名后的目录路径
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno
	
  作用：
    - 重命名目录
*/
```

**e.g.**

```c
/*
#include <stdio.h>

int rename(const char *oldpath, const char *newpath);
  参数：
	- oldpath: 要重命名的目录路径
    - newpath：重命名后的目录路径
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno
	
  作用：
    - 重命名目录
*/

#include <stdio.h>

int main() {

    int ret = rename("aaa", "bbb");

    if(ret == -1) {
        perror("rename");
        return -1;
    }

    return 0;
}
```



### d. `chdir`  和 `getcwd`

头文件：

```c
#include <unistd.h>
```



```c
int chdir(const char *path);
/*
  参数：
	- path : 需要修改的工作目录
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 修改进程的工作目录
*/
```

```c
char *getcwd(char *buf, size_t size);
/*
  参数：
	- buf: 存储的路径，指向的是一个数组（传出参数）
    - size: 数组的大小
	
  返回值：
	- 返回的指向的一块内存，这个数据就是第一个参数
	
  作用：
    - 获取当前工作目录
*/
```

```c
/*
#include <unistd.h>

int chdir(const char *path);
  参数：
    - path : 修改后的工作目录
	
  返回值：
	- 调用成功，返回0；调用失败，返回-1，并设置errno

  作用：
    - 修改进程的工作目录

char *getcwd(char *buf, size_t size);
  参数：
	- buf: 存储的路径，指向的是一个数组（传出参数）
    - size: 数组的大小
	
  返回值：
	- 返回的指向的一块内存，这个数据就是第一个参数
	
  作用：
    - 获取当前工作目录
*/

#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

int main() {

    // 获取当前的工作目录
    char buf[128];
    getcwd(buf, sizeof(buf));
    printf("当前的工作目录是：%s\n", buf);

    // 修改工作目录
    int ret = chdir("/home/nowcoder/Linux/lesson13");
    if(ret == -1) {
        perror("chdir");
        return -1;
    } 

    // 创建一个新的文件
    int fd = open("chdir.txt", O_CREAT | O_RDWR, 0664);
    if(fd == -1) {
        perror("open");
        return -1;
    }

    close(fd);

    // 获取当前的工作目录
    char buf1[128];
    getcwd(buf1, sizeof(buf1));
    printf("当前的工作目录是：%s\n", buf1);
    
    return 0;
}
```

---

## 13. 目录遍历函数

linux 系统中 “一切皆为文件”。所以，目录也是一个文件，其中存储这目录的信息和目录内子文件的信息，调用 `readdir` 函数可以逐条地读取目录中子文件的信息。

### a. `opendir`

头文件：

```c
#include <sys/types.h>
#include <dirent.h>
```



```c
DIR *opendir(const char *name);
/*
  参数：
   -  name：需要打开的目录名称

  返回值：
    - 调用成功，返回一个 DIR *（指针）指向一个目录流；调用失败，返回 NULL；

  作用：
    - 打开一个目录
*/
```

### b. `readdir`

头文件：

```c
#include <dirent.h>
```



```c
struct dirent *readdir(DIR *dirp);
/*
  参数：
    - dirp：存储 opendir 返回的结果

  返回值：
    - 调用成功，返回 struct dirent，其中存储了读取到的文件的信息（目录下的子文件，每次读取一项）；读取到末尾或者调用失败，返回 NULL

  作用：
    - 读取目录中的数据
*/
```



### c. `closedir`

头文件：

```c
#include <sys/types.h>
#include <dirent.h>
```



```c
int closedir(DIR *dirp);
/*
  参数：
    - dirp：要操作的目录的相关信息，由 opendir 获取

  返回值：
    - 调用成功，返回 0；调用失败，返回 -1

  作用：
    - 关闭与 dirp 关联的目录
*/
```

### d. `struct dirent`

```c
struct dirent
{
    ino_t d_ino; // 此目录进入点的inode
    off_t d_off; // 目录文件开头至此目录进入点的位移
    unsigned short int d_reclen; // d_name 的长度, 不包含NULL字符
    unsigned char d_type; // d_name 所指的文件类型
    char d_name[256]; // 文件名
};
```

### e. `d_type`

```c
d_type
DT_BLK - 块设备
DT_CHR - 字符设备
DT_DIR - 目录
DT_LNK - 软连接
DT_FIFO - 管道
DT_REG - 普通文件
DT_SOCK - 套接字
DT_UNKNOWN - 未知
```



### f. 举个例子

```c
/*
#include <sys/types.h>
#include <dirent.h>

// 打开一个目录
DIR *opendir(const char *name);
  参数：
   -  name：需要打开的目录名称

  返回值：
    - 调用成功，返回一个 DIR *（指针）指向一个目录流；调用失败，返回 NULL；

  作用：
    - 打开一个目录


#include <dirent.h>

struct dirent *readdir(DIR *dirp);
  参数：
    - dirp：存储 opendir 返回的结果

  返回值：
    - 调用成功，返回 struct dirent，其中存储了读取到的文件的信息（目录下的子文件，每次读取一项）；读取到末尾或者调用失败，返回 NULL

  作用：
    - 读取目录中的数据


#include <sys/types.h>
#include <dirent.h>

int closedir(DIR *dirp);
  参数：
    - dirp：要操作的目录的相关信息，由 opendir 获取

  返回值：
    - 调用成功，返回 0；调用失败，返回 -1

  作用：
    - 关闭与 dirp 关联的目录
*/

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>     // 
// #include <unistd.h>
#include <stdlib.h>

int getFileNum(const char * path);

int main(int argc, char * argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s <path>\n", argv[0]);
        return -1;
    }

    int num = getFileNum(argv[1]);

    printf("Total number of filse is: %d\n", num);
    
    return 0;
}

int getFileNum(const char * path)
{
    // 打开目录
    DIR * dir = opendir(path);
    if(dir == NULL)
    {
        perror("opendir");
        // return -1;
        exit(0); 	// 强制退出程序，直接返回操作系统
    }

    // 读取目录
    struct dirent *ptr;
    int tot = 0;
    while((ptr = readdir(dir)) != NULL)     // readdir 每次读取目录流中一条信息
    {
        char * dname = ptr->d_name;
        // printf("%s\n", dname);
        if(strcmp(dname, ".") == 0 || strcmp(dname, "..") == 0) continue;
        
        if(ptr->d_type == DT_DIR)   // 子目录
        {   
            char subPath[256] = {0};
            sprintf(subPath, "%s/%s", path, dname);
            tot += getFileNum(subPath);
        }
        else if(ptr->d_type == DT_REG)  // 普通文件
        {
            tot++;
        }
    }  
    // 关闭目录
    closedir(dir);

    return tot;
}
```

---



## 14. dup 和 dup2

### a. dup

```c
#include <unistd.h>
```

```c
int dup(int oldfd);
/*
  参数：
    - oldfd：要复制的文件描述符
  
  返回值：
    - 一个新的文件描述符（从空闲的文件描述符表中寻找一个最小的）

  作用：
    - 复制一个新的文件描述符。假设 oldfd 指向文件 a.txt，那么 fd2 = dup(oldfd) 也指向文件 a.txt
*/
```

### b. dup2

```c
#include <unistd.h>
```

```c
int dup2(int oldfd, int newfd);
/*
  参数：
    - oldfd：要指向的文件描述符
    - newfd：要重定向的文件描述符
  
  返回值：
    - 调用成功，返回 newfd；调用失败，返回 -1

  作用：
    - 重定向文件描述符。假设 oldfd 指向文件 a.txt，newfd 指向 b.txt，那么，调用 dup2(oldfd, newfd) 后，newfd 不再指向 b.txt，而是指向 a.txt
*/
```

### c. 举个例子

```c
/*
#include <unistd.h>

int dup(int oldfd);
  参数：
    - oldfd：要复制的文件描述符
  
  返回值：
    - 一个新的文件描述符（从空闲的文件描述符表中寻找一个最小的）

  作用：
    - 复制一个新的文件描述符。假设 oldfd 指向文件 a.txt，那么 fd2 = dup(oldfd) 也指向文件 a.txt

int dup2(int oldfd, int newfd);
  参数：
    - oldfd：要指向的文件描述符
    - newfd：要重定向的文件描述符
  
  返回值：
    - 调用成功，返回 newfd；调用失败，返回 -1

  作用：
    - 重定向文件描述符。假设 oldfd 指向文件 a.txt，newfd 指向 b.txt，那么，调用 dup2(oldfd, newfd) 后，newfd 不再指向 b.txt，而是指向 a.txt
*/

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

int main() {

    int fd = open("1.txt", O_RDWR | O_CREAT, 0664);
    if(fd == -1) {
        perror("open");
        return -1;
    }

    int fd1 = open("2.txt", O_RDWR | O_CREAT, 0664);
    if(fd1 == -1) {
        perror("open");
        return -1;
    }

    printf("fd : %d, fd1 : %d\n", fd, fd1);

    int fd2 = dup2(fd, fd1);
    if(fd2 == -1) {
        perror("dup2");
        return -1;
    }

    // 通过fd1去写数据，实际操作的是1.txt，而不是2.txt
    char * str = "hello, dup2";
    int len = write(fd1, str, strlen(str));

    if(len == -1) {
        perror("write");
        return -1;
    }

    printf("fd : %d, fd1 : %d, fd2 : %d\n", fd, fd1, fd2);

    close(fd);
    close(fd1);

    return 0;
}
```

---

## 15. fcntl

头文件：

```c
#include <unistd.h>
#include <fcntl.h>
```



```c
int fcntl(int fd, int cmd, ... );
/*
  参数：
    - fd：需要操作的文件描述符
    - cmd：表示对文件描述符的操作
      - F_DUPFD：复制文件描述符，返回一个新的文件描述符，和 fd 指向同一个文件。
        - e.g. int fd2 = fcntl(fd, F_DUPFD);
      - F_GETFL：获取（返回） fd 指向的文件的状态 flag，如O_RDONLY/O_APPEND 等；
      - F_SETFL：设置 fd 指向的文件的状态 flag（重置而非增加）
        - 必选项：O_RDONLY, O_WRONLY, O_RDWR 不可以被修改
        - 可选性：O_APPEND（追加数据）, O_NONBLOCK
                O_APPEND（设置成非阻塞）

  作用：
    - 基于文件描述符对文件进行操作
*/
```

举个例子：

```c
/*
#include <unistd.h>
#include <fcntl.h>

int fcntl(int fd, int cmd, ... );
  参数：
    - fd：需要操作的文件描述符
    - cmd：表示对文件描述符的操作
      - F_DUPFD：复制文件描述符，返回一个新的文件描述符，和 fd 指向同一个文件。int fd2 = fcntl(fd, F_DUPFD);
      - F_GETFL：获取（返回） fd 指向的文件的状态 flag，如O_RDONLY/O_APPEND 等；
      - F_SETFL：设置 fd 指向的文件的状态 flag（重置而非增加）
        - 必选项：O_RDONLY, O_WRONLY, O_RDWR 不可以被修改
        - 可选性：O_APPEND（追加数据）, O_NONBLOCK
                O_APPEND（设置成非阻塞）

  作用：
    - 基于文件描述符对文件进行操作
*/

#include <unistd.h>
#include <fcntl.h>

int main()
{
    int fd = open("1.txt", O_RDWR);
    if(fd == -1) {
        perror("open");
        return -1;
    }

    // 获取文件描述符状态flag
    int flag = fcntl(fd, F_GETFL);
    if(flag == -1) {
        perror("fcntl");
        return -1;
    }

    // 重置状态
    flag |= O_APPEND;

    // 修改文件描述符状态的flag，给flag加入O_APPEND这个标记
    int ret = fcntl(fd, F_SETFL, flag);
    if(ret == -1) {
        perror("fcntl");
        return -1;
    }

    char * str = "nihao";
    write(fd, str, strlen(str));

    close(fd);

    return 0;
}
```

<hr style="border:3px #6CA6CD double;">

# 进程

## 进程概述

- 进程是正在运行的程序的实例，是一个具有一定独立功能的程序关于某个数据集合的一次运行活动。它是操作系统动态执行的基本单元。在传统的操作系统中，进程既是基本的分配单元，也是基本的执行单元。
- 从内核的角度看，进程由用户内存空间和一系列内核数据结构组成，其中用户内存空间包含了程序代码以及代码所使用的的变量，而内核数据结构则用于维护进程状态信息。
- 单道程序设计和多道程序设计：
  - 单道程序设计：计算机内存中只允许一个程序运行。
  - 多道程序设计：计算机内存中同时包含多道相互独立的程序，在管理程序的控制下相互穿插运行。这些程序共享计算机系统资源，同处于开始到结束之间的状态。引入多道程序设计技术的根本目的是提高 CPU 的利用率。
  - 无论是单道程序设计还是多道程序设计，就微观而言，单个 CPU 上运行的进程只有一个。
- 时间片：操作系统分配给每个正在运行的进程微观上的一段 CPU 时间。时间片由操作系统内核的调度程序分配给每个进程。首先，内核会给每个进程分配相等 的初始时间片，然后每个进程轮番地执行相应的时间，当所有进程都处于时间片耗尽的 状态时，内核会重新为每个进程计算并分配时间片，如此往复。
- 并发和并行：
  - 并发：在同一时刻只有一条指令执行。宏观上具有多个进程同时执行的效果，但在微观上并不是同时执行的， 只是把时间分成若干段，使多个进程快速交替的执行。
  - 并行：在同一时刻，有多条指令在**多个处理器**上同时执行。
- 为了管理进程，内核为每个进程分配一个 PCB (Processing Control Block) 进程控制块。Linux 内核的进程控制块是 task_struct 结构体。（/usr/src/linux-headers-xxx/include/linux/sched.h）



## 进程状态

进程状态反映进程执行过程的变化。这些状态随着进程的执行和外界条件的变化而转换。 在三态模型中，进程状态分为三个基本状态，即就绪态，运行态，阻塞态。在五态模型 中，进程分为新建态、就绪态，运行态，阻塞态，终止态。

<img src="./assets/image-20231102181911419.png" alt="image-20231102181911419" style="zoom: 75%;" />

- 运行态：进程占有处理器正在运行。
- 就绪态：进程具备运行条件，等待系统分配处理器以便运 行。当进程已分配到除CPU以外的所有必要资源后，只要再 获得CPU，便可立即执行。在一个系统中处于就绪状态的进 程可能有多个，通常将它们排成一个队列，称为就绪队列。
- 阻塞态：又称为等待(wait)态或睡眠(sleep)态，指进程 不具备运行条件，正在等待某个事件的完成。

<img src="./assets/image-20231102181928295.png" alt="image-20231102181928295" style="zoom:67%;" />

- 新建态：进程刚被创建时的状态，尚未进入就绪队列。
- 终止态：进程完成任务到达正常结束点，或出现无法克服的错误而异常终止，或被操作系统及 有终止权的进程所终止时所处的状态。进入终止态的进程以后不再执行，但依然保留在操作系 统中等待善后。一旦其他进程完成了对终止态进程的信息抽取之后，操作系统将删除该进程。

## 进程号

- 每个进程都由进程号来标识，其类型为 pid_t（整型），进程号的范围：0～32767。 进程号总是唯一的，但可以重用。当一个进程终止后，其进程号就可以再次使用。
- 任何进程（除 init 进程）都是由另一个进程创建，该进程称为被创建进程的父进程， 对应的进程号称为父进程号（PPID）。
- 进程组是一个或多个进程的集合。他们之间相互关联，进程组可以接收同一终端的各 种信号，关联的进程有一个进程组号（PGID）。默认情况下，当前的进程号会当做当 前的进程组号。

## 进程相关命令

### 1. ps aux / ajx

- a：显示终端上的所有进程，包括其他用户的进程 
- u：显示进程的详细信息 
- x：显示没有控制终端的进程 
- j：列出与作业控制相关的信息

<img src="./assets/image-20231102182223895.png" alt="image-20231102182223895" style="zoom:67%;" />

### 2. top

可以在使用 top 命令时加上 -d 来指定显示信息更新的时间间隔，在 top 命令 执行后，可以按以下按键对显示的结果进行排序： 

- M - 根据内存使用量排序 
- P - 根据 CPU 占有率排序 
- T - 根据进程运行时间长短排序 
- U - 根据用户名来筛选进程 
- K - 输入指定的 PID 杀死进程

### 3. kill

- kill [-signal] pid 
- kill –l 列出所有信号 
- kill –SIGKILL 进程ID 
- kill -9 进程ID

ulimit -a



### tty 

查看当前终端ID



## 进程创建

### 1. 父子进程虚拟地址空间

- 创建的子进程，会**在分离的存储空间中创建一个父进程的副本（包括用户区和内核区）。即，深拷贝**。内核区中的 pid 为各自的进程号。`fork` 函数的返回值保存在栈空间中。
- 如果在创建子进程之前，父进程中有一个变量 `num`，那么创建的子进程中也会有这样一个变量 `num`，但是两个 `num` 毫无关系，父进程中对 `num` 的操作不会影响子进程中的 `num`。

<img src="./assets/image-20231103133637108.png" alt="image-20231103133637108" style="zoom:100%;" />

### 2. fork

- 一个现有的进程可以调用 `fork` 函数创建一个新进程，称为子进程。`fork` 函数被调用一次，但**返回两次**，一次返回给子进程，返回值为0，一次返回给父进程，返回值为子进程的进程号。（因为父子进程是一对多的关系，所以子进程返回 0 即可，而父进程需要返回子进程的进程号）。
- 当该子进程创建时，它和父进程都会 **从 `fork` 调用的下一条（或者说从 `fork` 的返回处）**开始执行继续执行与父进程相同的代码。
- 子进程是父进程的副本，准确的说，子进程获取父进程 **数据空间、堆和栈的副本**。**父进程和子进程不共享这些存储空间，但是父子进程共享正文段**。

- 准确的说，`fork` 是通过写时拷贝（copy-on-write，推迟或者避免拷贝数据的技术）实现的。内核在刚创建子进程时不会拷贝整个父进程的地址空间，而是父子进程共享同一个地址空间。只有当父/子进程出现了写入操作时，内核才会真正地进行拷贝，将父子进程的地址空间分离。**因为 `fork` 之后经常跟随 `exec`，所以现在很多实现都不执行父进程数据段、栈、堆的完全副本**。

- 父进程在 `fork` 之前打开的文件描述符，都会被拷贝给子进程。父进程和子进程共享同一个文件偏移量，也就是说，当父子进程向同一个文件（`fork` 之前打开的）写时，父子进程会相互影响。因此，在 fork 之后通常这样处理文件描述符：

  - 父进程等待子进程完成；
  - 父进程和子进程各自关闭他们不需要使用的文件描述符，避免干扰对方。

  <img src="./assets/image-20231105145200619.png" alt="image-20231105145200619" style="zoom:80%;" />

- 父进程和子进程会交替进行，但这并不意味着一定是轮流进行，也可能是子进程执行好几个时间片后再切换为父进程。
- `fork` 之后，父子进程谁先执行也是不确定的，这取决于内核使用的调度算法。



头文件：

```c
#include <sys/types.h>
#include <unistd.h>
```



```c
pid_t fork(void);
/*
  参数：
    - None

  返回值：
    - 返回两次（分别在父子进程中各返回一次）。在父进程中返回子进程的进程号，在子进程中返回 0。可以通过返回值区分父子进程。
    - 若创建子进程失败，会在父进程中返回 -1，并设置 errno。
      - errno 设为 EAGAIN：当前系统的进程数已经达到系统规定的上限；
      - errno 设为 ENOMEN：系统内存不足

  作用：
    - 创建子进程
      - 父进程希望复制自己，使父子进程同时执行不同的代码段。如网络服务进程中父进程等待客户端的服务请求：当请求到达时，父进程调用 fork 使进紫禁城处理此请求，父进程则继续等待下一个服务请求；
      - 一个进程要执行一个不同的程序。如，子进程从 fork 返回后立即调用 exec。（spawn）
*/
```



举个例子：

```c
/*
#include <sys/types.h>
#include <unistd.h>

pid_t fork(void);
  参数：
    - None

  返回值：
    - 返回两次（分别在父子进程中各返回一次）。在父进程中返回子进程的进程号，在子进程中返回 0。可以通过返回值区分父子进程。
    - 若创建子进程失败，会在父进程中返回 -1，并设置 errno。
      - errno 设为 EAGAIN：当前系统的进程数已经达到系统规定的上限；
      - errno 设为 ENOMEN：系统内存不足

  作用：
    - 创建子进程
*/

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main()
{
    // create a new pid
    pid_t pid = fork();
    pid_t child, parent;

    if(pid > 0)
    {
        // parent
        printf("pid: %d\n", getpid());
        parent = getpid();
        printf("parent pid: %d, ppid: %d\n", getpid(), getppid());
    }
    else if (pid == 0)
    {
        //child
        child = getpid();
        printf("child pid: %d, ppid: %d\n", getpid(), getppid());
    }
    for(int i = 0; i < 5; i++)
    {
        if(getpid() == parent)
        {
            printf("parent i: %d, pid: %d\n", i, getpid());
        }
        else if(getpid() == child)
        {
            printf("child i: %d, pid: %d\n", i, getpid());
        }
        sleep(1);
    }

    return 0;
}
```

运行结果：

<img src="./assets/image-20231103112248434.png" alt="image-20231103112248434" style="zoom:80%;" />

调用 `ps aux` 命令查看进程 10067，发现这是一个终端（终端是一个阻塞的进程，等待用户交互）：

![image-20231103112357328](./assets/image-20231103112357328.png)

调用 `tty` 命令查看当前终端 ID，相同：

<img src="./assets/image-20231103112424559.png" alt="image-20231103112424559" style="zoom:80%;" />

---

## 3. exec 函数族

- 当进程调用一种 `exec` 函数时，该进程执行的程序完全替换为新程序，而 **新程序则从其 `main` 函数开始执行**。**`exec` 函数用磁盘上的一个新程序替换了当前进程的正文段、数据段、堆段和栈段。**

  <figure class="half">
      <img src="./assets/image-20231105162349837.png" alt="image-20231105162349837" style="zoom:70%;" />
      <img src="./assets/image-20231105162657868.png" alt="image-20231105162657868" style="zoom:80%;" />
  </figure>

- 当内核执行 exec 调用时，在调用 main 前会先调用一个特殊的启动例程，可执行程序文件将此启动例程指定为程序的起始地址。启动例程从内核去的命令行参数和环境变量值，为调用 main 函数做好安排。







### a. execl

```c
int execl(const char *pathname, const char *arg, ... /* (char  *) NULL */);
/*
  参数：
    - pathname：要执行的程序路径（相对路径或绝对路径）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾）

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```



```c
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>

int main()
{
    // create a new process
    pid_t pid = fork();
    pid_t child, parent;
    if(pid > 0)
    {
        // parent process
        printf("pid: %d\n", getpid());
        parent = getpid();
        printf("parent pid: %d, ppid: %d\n", getpid(), getppid());
    }
    else if(pid == 0)
    {
        // child process
        execl("hello", "hello", NULL); 	// 执行 hello 可执行文件（输出 “hello world”）
        // execl("/bin/ps", "ps", "aux", NULL); 	// 执行系统调用
        child = getpid();
        printf("child pid: %d, ppid: %d\n", getpid(), getppid());
    }
    for(int i = 0; i < 3; i++)
    {
        if(getpid() == parent)
        {
            printf("parent i: %d, pid: %d\n", i, getpid());
        }
        else if(getpid() == child)
        {
            printf("child i: %d, pid: %d\n", i, getpid());
        }
        sleep(1);
    }
    return 0;
}
```

运行结果：

子进程仅执行 hello。

![image-20231105165733051](./assets/image-20231105165733051.png)

子进程仅执行 `ps aux` 命令

![image-20231105165818494](./assets/image-20231105165818494.png)

### b. execlp

```c
int execlp(const char *file, const char *arg, ... /* (char  *) NULL */);
/*
  参数：
    - file：要执行的可执行文件路径（自动在系统环境变量下搜索可执行文件）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾）

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```

```c
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>

int main()
{
    // create a new process
    pid_t pid = fork();
    pid_t child, parent;
    if(pid > 0)
    {
        // parent process
        printf("pid: %d\n", getpid());
        parent = getpid();
        printf("parent pid: %d, ppid: %d\n", getpid(), getppid());
    }
    else if(pid == 0)
    {
        // child process
        execlp("ps", "ps", "aux", NULL); 	// 自动在环境变量中寻找 ps 可执行文件
        // execlp("/home/ubuntu/lyy/cpp/webserver/hello","hello", NULL); 	// 绝对路径仍可以执行
        child = getpid();
        printf("child pid: %d, ppid: %d\n", getpid(), getppid());
    }
    for(int i = 0; i < 3; i++)
    {
        if(getpid() == parent)
        {
            printf("parent i: %d, pid: %d\n", i, getpid());
        }
        else if(getpid() == child)
        {
            printf("child i: %d, pid: %d\n", i, getpid());
        }
        sleep(1);
    }
    return 0;
}
```



### c. execle

```c
int execle(const char *pathname, const char *arg, ... /*, (char *) NULL, char *const envp[] */);
/*
  参数：
    - pathname：要执行的可执行文件路径（在自己定义的环境路径下搜索可执行文件）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾）
    - envp：自己定义的环境路径，如char * envp[] = {"/home/ubuntu", "home/bin"};

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```



### d. execv

```c
int execv(const char *pathname, char *const argv[]);
/*
  参数：
    - pathname：要执行的程序路径（相对路径或绝对路径）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾），以数组的形式传入，如， char * argv[] = {"ps", "aux", NULL};

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```

```c
int execvp(const char *file, char *const argv[]);
/*
  参数：
    - file：要执行的可执行文件路径（自动在系统环境变量下搜索可执行文件）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾），以数组的形式传入，如， char * argv[] = {"ps", "aux", NULL};

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```

```c
int execvpe(const char *file, char *const argv[], char *const envp[]);
/*
  参数：
    - pathname：要执行的可执行文件路径（在自己定义的环境路径下搜索可执行文件）
    - arg：执行程序需要的参数（第一个为可执行文件的名称，中间为可执行文件需要的参数，最后需要以 NULL 结尾），以数组的形式传入，如， char * argv[] = {"ps", "aux", NULL};
    - envp：自己定义的环境路径，如char * envp[] = {"/home/ubuntu", "home/bin"};

  返回值：
    - 调用失败，返回-1，并设置errno；调用成功，没有返回值（因为程序段已经被替换，不会再回到原 main）
*/
```

---





































==c不支持函数重载==

