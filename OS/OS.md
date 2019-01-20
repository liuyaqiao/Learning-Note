- 地址

物理地址：

计算机主存储器由M个连续的字节大小单元组成的数组。每一个字节都有一个唯一的物理地址。CPU访问内存最直接的方式就是直接访问物理地址。

虚拟地址

虚拟地址是一个类似于偏移量的非实体地址。在使用虚拟寻址的时候，我们会使用地址翻译先将虚拟地址转换成物理地址。这也需要CPU硬件和OS的结合。

- context switching

这是一种较高形式的异常控制流来实现多任务。

1. how to fit lots of program into the memory

因为他们有不同的地址，但是如何去fit。使得每一个program都可以用一段合适的地址。所以我们给他们不同而且分离的物理地址，这样的话，我们不能直接访问或者实现切换。

- 可以pre-divide
- hardware address translation (virtual addr and physical addr)

大体思路：

我们要使用interrupt instructions 进行system calls到supervise mode去调用另一个base register。这是我们切换process的基本思路，下一步我们要讨论这是如何工作的：

2. how to switch program 

这时我们要讨论context switching。

yield12 ():   这是函数的基本形式
    context_switch(&sp1,sp2)的形式
    
it leaves a processor thread by calling and it enters a process processor thread by returning.

return to enter previous process
call to leave current process

我们有一个timer去控制context switching，每隔一定时间就会调用interrupt，去进行context switching。这个函数return和enter stack的位置不同，return会返回另一个进程。



<liu.yaqi@husky.neu.edu>

