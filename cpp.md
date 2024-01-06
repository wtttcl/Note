1. 声明和定义的区别

   - 声明：向编译器宣告标识符（如变量、函数、类等）的存在和类型，但不分配内存或者不提供具体实现。声明告诉编译器，“这个东西在某处存在，你将在后面找到它的定义”。

     ```c++
     extern int x;
     ```

     

   - 定义告诉编译器，“这个东西在这里真正被创建了”。

     ```c++
     int x = 5;
     ```

2. 要想 **建立在整个类中都恒定的常量**，需要用 `static const`。

   在 C++ 中：

   - 静态成员变量是 **类的所有对象共有的**，**不属于任何一个类对象**。==静态成员变量被存储在栈中（？）==。
   - 静态成员变量必须在 **类外定义**，在 **类内声明**，且定义时不用添加 `static` 关键字，但要指明属于哪个类。
   - 要想 **建立在整个类中都恒定的常量**，需要用 `static const`。

   ```c++
   class Solution {
   private:
       static const int HIGH_BIT = 30;
   
   public:
       int findMaximumXOR(vector<int>& nums)
       {
           for (int k = HIGH_BIT; k >= 0; --k)
           {...}
   		...
       }
   };
   
   ```

3. ==类内成员变量，static 和 const 和 static const 区别。==





1. string.h cstring string 的区别



## 定义和初始化



## 
