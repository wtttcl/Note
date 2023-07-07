### 计算机硬件组成

计算机硬件组成：输入设备、输出设备、控制器、运算器、存储器

- CPU：运算器 + 控制器

- 存储器 ：内存（临时存储） + 硬盘（永久存储）

- 输入设备：鼠标 + 键盘

- 输出设备：显示器 + 打印机

### 计算机软件

系统软件：操作系统（windows、linux、mac）

应用软件可以被分为两种形式，

#### 应用软件

C/S：client/server 客户端/服务器端

在用户本地有一个客户端程序，在远程有一个服务器端程序（QQ、微信）

B/S：browser/server浏览器/服务器端

只需要一个浏览器，用户通过不同网址，访问不同的服务器端程序（京东、淘宝）

### 计算机语言

机器语言：0/1代码

汇编语言

高级语言

### 人机交互

- 图形化界面

- 命令行

  （DOS命令行）win + r -> cmd:

  ```shell
  shutdown -s -t 300 	# 在300s后关机
  shutdown -a 	# 取消关机任务
  
  dir 	# 展示当前目录下文件和文件夹
  cd/ 	# 直接返回盘目录
  cls 	# 清屏
  exit 	# 退出命令行
  ```

### Java版本

- Java SE：

  Java语言的标准版，用于桌面应用开发，是其他两个版本的基础。

  桌面应用：用户只要打开程序，程序的界面会让用户在最短的时间内找到他们需要的功能，同时主动带领用户完成他们的工作并得到最好的体验。

- Java ME：

  Java语言的小型版，用于嵌入式消费类电子设备。

- Java EE：

  Java语言的企业版，用于Web方向的网站开发。

  网页：通过浏览器将 **数据展示** 在用户面前，跟后台服务器没有交互

  网站：通过跟 **后台服务器的交互** ，将查询到的真实数据再通过网页展示出来。网站 = 网页 + 后台服务器。

### Java跨平台工作原理

平台：操作系统（windows、linux、mac）

跨平台：Java程序可以在任意操作系统上运行。

Java跨平台工作原理：Java本质上是运行在操作系统的 **JVM虚拟机** 上，只要操作系统安装了 JVM虚拟机 即可。**ps. JVM虚拟机本身不允许跨平台，允许跨平台的是Java程序。**

### JRE 和 JDK

JRE： Java Runtime Environment，指Java运行环境，**包括 JVM虚拟机**以及 Java 核心类库（一个java文件就是一个类，类库指存放多个 java 文件的仓库）。

JDK：Java Development Kit，是Java语言的软件开发工具包，内部包含了 **JRE** 和代码的 **编译工具** 和 **运行工具** 等。

![image-20230625144651179](./assets/image-20230625144651179.png)

![image-20230703104151201](./assets/image-20230703104151201.png)

#### JRE、JDK、JVM的作用

- 编写代码的过程中，需要使用 JRE 中 Java已经写好的代码
- 编译代码的过程中，需要使用 JDK 中的翻译工具。
- 运行代码的过程中，需要使用 JDK 中的运行工具。
- 代码需要运行在 JVM 当中。

#### JDK、JRE、JVM的关系

![image-20230625144744663](./assets/image-20230625144744663.png)

## JDK 安装

bin 目录下存放了 JDK 的各种工具命令，javac 和 java 就放在这个目录里。



## 使用 javac 和 java 实现 HelloWorld

ATTENTION: 要使用 javac 和 java 编译和执行 java 文件时，java 文件必须和他们在同一目录下。

javac：编译工具，命令行执行

```shell
javac xxx.java
```

java：运行工具，命令行执行

```
java xxx
```

![image-20230703105609984](./assets/image-20230703105609984.png)

**ps. 若代码中没有 main 方法，则无法运行。**

## path 环境变量

能够在任意目录下访问 bin 目录下的 javac 和 java。

![image-20230703110313602](./assets/image-20230703110313602.png)

**ps. 不一定要叫 JAVA_HOME，但是在某些软件中只能识别 JAVA_HOME，所以命名为 JAVA_HOME 比较方便。**

![image-20230703110440897](./assets/image-20230703110440897.png)

![image-20230703110454712](./assets/image-20230703110454712.png)

## 注释

单行注释：//

多行注释：/**/



## IDEA

IDEA，全称 IntelliJ IDEA，适用于 java 语言开发的集成环境。

集成环境：把代码编写、编译、执行、调试等多重功能综合到一起的开发工具。

### 结构

- project：项目、工程
- module：模块
- package：包
- class：类

### 类文件的修改

添加：右键 - refactor

删除：右键 - delete **（无法在回收站找回）**

### 模块的修改

新建：file - project structure

修改：右键 - refactor - rename module and directory

移除：右键 - reomve module**（只在列表中移除，仍在硬盘中，即文件夹不会被删除）**

导入：file - project structure**（导入模块后会报错 --》 点积右上角 setup SDK（因为没有和本地 SDK 关联））**

### 项目的修改

修改：file-program structure **（修改名字，不会修改路径，要修改路径，需要在关闭项目后rename文件夹再打开）**



### Package

包在硬盘中是一个文件夹

com.itheima --> com文件夹下itheima文件夹



## 注释TODO

标记代码，表示待完成或待解决的部分。

```java
// TODO:
```



# 基础知识

## 面向对象的三大特征

### 封装

隐藏实现细节，仅对外暴露公共的访问方式。

常见形式：

- 将代码抽取到方法中（对代码封装）
- 将属性抽取到类中（对数据封装）



### 继承

### 多态



## 关键字

被 java 赋予特定含义的英文单词，全部小写。



## 常量

字符串常量（双引号）、整数常量、小数常量、字符常量（单个字符/单个汉字，单引号）、布尔常量（true/false）、空常量（null，不能直接打印）。



### 进制书写格式（jdk7版本后才支持）

- 十进制，默认
- 二进制，0b/0B开头
- 八进制，0开头
- 十六进制，0x/0X开头



## 变量（某内存空间的别名）

### 定义格式（需要被初始化，否则使用时会报错）

```java
数据类型 变量名 = 数据值；
```



## 标识符（自己起的变量名）

### 命名规则

- 数字、字模、下划线(_)、美元符($)
- 不能以数字开头
- 不能是关键字
- 区分大小写

### 命名规范

- 小驼峰命名法：**（方法、变量）**
  - 标识符是一个单词，首字母小写，e.g. name
  - 标识符是多个单词，除第一个单词外的单词首字母大写，e.g.firstName

- 大驼峰命名法：**（类）**
  - 标识符是一个单词，首字母大写，e.g. Name
  - 标识符是多个单词，所有单词首字母大写，e.g.FirstName

## 数据类型

### 基本数据类型

- 整数
  - byte : 8 (-128 ~ 127)
  - short : 16 (-32768 ~ 32767)
  -  **int**（默认，首选）: 32 (-2147483648 ~ 2147483647)
  -  long : 64 (-922337036854775808 ~ 922337036854775807) （定义变量时，要在末尾加上 **L** 标识，建议大写，小写太像1了）
- 浮点数
  - float : 32 （定义变量时，要在末尾加上 **F** 标识，大小写都行）
  - **double**（默认，首选）: 64
- 字符
  - char : 8 (0 ~ 65535) （也可直接赋值为 ASCII 码）
- 布尔
  - boolean : true / false

### 引用数据类型

引用/记录地址值的变量，包括：类的对象、数组、接口。



## 键盘录入变量值

```java
import java.util.Scanner;

Scanner sc = new Scanner(Syetem.in);

int age = sc.nextInt(); 	// 录入int
String name = sc.next(); 	// 录入字符串
```



## 运算符



### 算术运算符

- +
- -
- *
- /：整数相除，只返回商/整数；有小数则返回小数
- %：取余



#### 字符串的拼接

- 当 "+" 操作中有一边出现字符串时，"+" 实现的是字符串拼接操作，如
  -  "item" + 666 = "item666"
- 但要注意表达式是 **从左往右** 运算的，所以
  - 1 + 99 + "年黑马" = "100年黑马"



### 自增（++）自减（--）运算符

运算符在前，变量先完成自增长，然后再参与运算；

运算符在后，先完成运算，再实现自增长。

ps. 只能操作变量，不能操作常量，因为 常量 = 常量 + 1 不合理

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int a = 10;
        int b = a++;
        System.out.println(a);  // 11
        System.out.println(b);  // 10
    }
}
```

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int a = 10;
        int b = ++a;
        System.out.println(a);  // 11
        System.out.println(b);  // 11
    }
}
```





### 赋值运算符

- =
- +=
- -=
- *=
- /=
- %=

ps. 后五个扩展赋值运算符，自带强制类型转换效果。

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int a = 10;
        byte b = 20;

        b = (byte)(a + b);
        System.out.println(b);  // 30
    }
}
```

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int a = 10;
        byte b = 20;

        b += a;
        System.out.println(b);  // 30
    }
}
```



### 关系运算符

- ==
- !=
- <
- <=
- (>)
- (>=)



### 逻辑运算符（没有短路效果）

- &
- |
- ！
- ^ ：异或



### 短路逻辑运算符

- &&
- ||

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int x = 3, y = 4;

        System.out.println(++x > 4 & y-- < 5);  // false & true = false
        System.out.println("x=" + x);   // 4
        System.out.println("y=" + y);   // 3
    }
}
```

```java
public class ZiZengZiJian {
    public static void main(String[] args) {
        int x = 3, y = 4;

        System.out.println(++x > 4 && y-- < 5);  // false
        System.out.println("x=" + x);   // 4
        System.out.println("y=" + y);   // 4
    }
}
```



### 三元运算符

格式：

```java
expression ？ value1 ： value2;
```

```java
public class Test {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("Print three numbers:");

        int num1 = sc.nextInt();
        int num2 = sc.nextInt();
        int num3 = sc.nextInt();

        int tmp = num1 > num2 ? num1 : num2;
        int ans = num3 > tmp ? num3 : tmp;

        System.out.println("The biggest number is " + ans);
    }
}
```



## 流程控制语句

### 顺序结构

默认的执行顺序

### 分支结构

#### if 语句

```java
if(judge1)
{
    // ...
}
else if(judge1)
{
    // ...
}
else
{
    // ...
}
```



#### switch 语句

```java
switch(expression)
{
    case value1:
    	// ...
    	break;
    case value2:
    	// ...
    	break;
    // ...
    default: 	// 没有default，此外 vlaue 不执行任何语句
    	// ...
    	break;
}
```



ps. case 后只能是常量，且不能重复

## 循环结构



# 类型转换

### 隐式转换

- 把一个取值范围小的数值或变量赋值给另一个取值范围大的变量，会隐式转换成取值范围大的类型。

  ```java
  public class ZiZengZiJian {
      public static void main(String[] args) {
          int a = 10;
          double b = a;
          System.out.println(a);  // 10
          System.out.println(b);  // 10.0
      }
  }
  ```

  

![image-20230705104134070](./assets/image-20230705104134070.png)

- 取值范围小的数据和取值范围大的数据进行运算时，小的会先隐式转换成大的，再进行运算。

  ![image-20230705104236990](./assets/image-20230705104236990.png)

  

- byte / short / char 三种数据在运算时，会先隐式转换成 int，再进行运算。因此，下述代码会报错：

  ![image-20230705104358423](./assets/image-20230705104358423.png)

### 强制转换

- 把一个取值范围大的数值或变量赋值给另一个取值范围小的变量时，不允许直接赋值，需要加入强制转换。

  格式：

  ```java
  typeName tar_var = (typeName) var;
  ```



# 类和对象




## 举个例子：四句代码出窗体

```java
import javax.swing.*;

public class App {
    public static void main(String[] args) {
        JFrame frame = new JFrame(); 	// IDEA 识别到这条语句时会自动导包

        frame.setSize(514,538);     // 设置窗体长宽

        frame.setLocationRelativeTo(null);  // 居中

        frame.setAlwaysOnTop(true);     // 置顶

        frame.setDefaultCloseOperation(3);  //关闭模式，关闭窗体即结束程序

        frame.setTitle("2048 小游戏");     //设置窗体 title

        frame.setVisible(true);     // 设置窗口可见，这条语句要放在最后执行
    }
}
```



## 类（提高代码的复用性和可执行性）

类的组成：属性（成员变量） + 行为（成员方法）



### Example

```java
public class Student {
    String name;
    int age;

    public void show(){
        System.out.println("====== student ======");
    }
}
```

```java
public class TestStudent {
    public static void main(String[] args) {
        Student stu = new Student();
        System.out.println(stu);    // com.itheima.Student@b4c966a （对象在内存中的地址值）

        /* 成员变量没有初始化也可以直接使用，但使用的是默认初始化值。
            整数：0
            小数：0.0
            布尔：false
            字符：'\u0000' （空白字符）
            引用数据类型：null
        */
        System.out.println(stu.name);   // null
        System.out.println(stu.age);    // 0

        stu.name = "xxx";
        stu.age = 23;
        System.out.println(stu.name + "---" + stu.age);     // xxx---23

        stu.show();     // ====== student ======
    }
}
```



### 成员变量和局部变量的区别

![image-20230706145431086](./assets/image-20230706145431086.png)



### 成员变量和局部变量重名 - 就近原则

```java
public class Student {
    String name;
    int age;

    public void sayHello(String name){
        System.out.println(name + this.name); 	// name 就近选择 String name
    }
    
    public void sayHello(){
        System.out.println(name); 	// this. 可以省略，java内部自动补全
    }
}

public class TestStudent {
    public static void main(String[] args) {
        Student stu = new Student();
        stu.name = "xxx";
        stu.age = 23;
        
        stu.sayHello("yyy"); 	// yyyxxx
        stu.sayHello(); 	// xxx
    }
}
```



### this 关键字 

- 指向当前对象，this 代表当前对象的引用，值为当前对象的地址。
- 可调用对象的成员变量和成员方法。

- 区分局部变量和成员变量的重名。

![image-20230707131609869](./assets/image-20230707131609869.png)



### 权限修饰符

- public：同一个类、同一个包、**不同的包**。
- private：只能在类中访问。
- (default)：默认，在同一个类中、**同一个包**下进行访问。
- protected：



### 方法重载

Java 虚拟机会通过 **参数的不同** 来 **区分同名方法**。

ps. 同样地参数类型，但是不用的顺序也会构成重载，但是意义不大。



### 构造方法

- 在创建对象时对成员变量进行初始化，可以重载。
- java 提供一个默认的、无参数的构造方法。但是如果显式定义了构造方法，系统将不再提供默认的构造方法，因此，建议 **显式定义无参数的和有参数的构造方法**。

```java
public class Student {
    String name;
    int age;
	
    public Student()
    {
        
    }
    
    public Student(String name, int age)
    {
        this.name = name;
        this.age = age;
        System.out.println("this is the constructor of class student.");
        System.out.println("name: " + this.name + "    age: " + this.age);
    }
}

public class TestStudent {
    public static void main(String[] args) {
        Student stu = new Student("xxx", 18);
    }

}

/*
this is the constructor of class student.
name: xxx    age: 18
*/
```



![image-20230707133532215](./assets/image-20230707133532215.png)

### 标准的 JavaBean 类

- 成员变量使用 private 修饰
- 构造方法提供一个无参的和一个带参的
- 成员方法中要提供每个成员变量对应的 setXxx() 和 getXxx() 

```java
public class Student {
    private String name;
    private int age;

    public Student(){}

    public Student(String name, int age)
    {
        this.name = name;
        this.age = age;
    }

    public void setName(String name)
    {
        this.name = name;
    }
    public String getName()
    {
        return name;
    }

    public void setAge(int age)
    {
        this.age = age;
    }
    public int getAge()
    {
        return age;
    }
}
```



```java
public class TestStudent {
    public static void main(String[] args) {
        Student stu = new Student();
        stu.setName("xxx");
        stu.setAge(18);
        System.out.println(stu.getName() + stu.getAge());

        Student stu2 = new Student("xxx", 18);
        System.out.println(stu2.getName() + stu2.getAge());
    }
}
```



### 对象作为形参在方法中传递

对象在方法中作为参数传递时，传递的本质上是记录该对象的地址。

![image-20230707135328149](./assets/image-20230707135328149.png)

# Java 内存模型

- 堆内存：保存程序运行时产生的对象
- 方法区：保存方法。
- 栈内存：保存程序运行时所有方法执行状态（ **包括方法中产生的对象** ）。方法被执行后就会进入栈内存。
- 本地方法栈：管理一些特殊方法
- 寄存器

![image-20230706144202745](./assets/image-20230706144202745.png)

### 方法调用过程

![image-20230706145120462](./assets/image-20230706145120462.png)



### 带参数的成员方法调用过程



![image-20230706150523215](./assets/image-20230706150523215.png)

### 带返回值的成员方法调用过程

![image-20230706150835984](./assets/image-20230706150835984.png)























## 快捷方式

- pvsm + enter : main方法

- sout + enter : 输出语句 

- ctrl + D : 向下复制一行

- 自动生成构造函数和 setXxx / GetXxx

  ![image-20230707134850244](./assets/image-20230707134850244.png)

![image-20230707134933820](./assets/image-20230707134933820.png)

is ok？
