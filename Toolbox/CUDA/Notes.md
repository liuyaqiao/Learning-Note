# GPU Programming

## CUDA API

![api](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/Toolbox/CUDA/cudaapi.png)

## CUDA concept：

software：
1. host device kernel complier
2. 执行cuda程序的过程
3. cuda中grid、block、thread结构

## GPU Architecture：
hardware：
![GPU](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/Toolbox/CUDA/kepler%20GPU.png)

SM:
1. Control unit
    - warp schdedular
    - instruction dispatcher
2. Execution unit
    - CUDA cores(SP)
    - Speical Function Unit, LD/ST(Load Store Unit)
3. memory
    - Registers
    - Cache
        - L1
        - texture cache
        - shared memory
        - constant cache
        
SM的数量决定了计算能力。

SP：最基本的处理单元，streaming processor，也称为CUDA core。最后具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。

多个SP加上其他的一些资源组成一个streaming multiprocessor。也叫GPU大核，其他资源如：warp scheduler，register，shared memory等。SM可以看做GPU的心脏（对比CPU核心），register和shared memory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。**这也是优化的方向。**

在kepler中，一个逻辑控制单元控制32个cores，避免了因为逻辑控制单元太少而造成的cuda核心的冗余。

## Memory Hierarchy:

CUDA构架中，最小的执行单元是thread。数个thread可以组成一个block，一个block中的thread能够读取同一块内存空间，这里就是share memory，他们可以快速进行同步工作。每一个block所能包含的thread的数目是有限的，不过执行相同指令的block可以构成一个grid。一个grid执行的是一个kernel函数。

每个 thread 都有自己的一份 register 和 local memory 的空间。同一个 block 中的每个 thread 则有共享的一份 share memory。此外，所有的 thread(包括不同 block 的 thread)都 共享一份 global memory、constant memory、和 texture memory。不同的 grid 则有各自的 global memory、constant memory 和 texture memory。

我们最好的情况是需要高带宽，低延迟的情况；

- share memory

在share memory中，所有的数据以bank的数据存储，它可以分钟16个bank。如果不同的thread存取不同的bank，则不会有问题。如果他们读取同一个bank，则会发生bank conflict的问题，这时thread必须按照顺序去读取，无法同时读取shared memory。bank以4字节为单位，分成bank。

这里需要考虑的问题有两个：

1. memory的储存方式
2. warp的读取方式会不会造成bank conflict

- Global memory

对于global memory，因为它的读取延迟比较高。我们要考虑的是读取的内容要尽可能的连续，这就是coalesced的要求。初次之外，它开始的地址必须是每一个thread所存取大小的16倍。

## CUDA Program

1. kernel 函数

thread :
- all threads execute same sequential program
- threads execute in parallel

thread block : a group of threads
- executes on a single SM
- threads within a block can cooperate
    - light weight synchronization
    - data exchange - shared memory

thread grid
- thread blocks of a grid execute across multiple SMs
- thread blocks do not synchronize with each other
- Communicatication between blocks is expensive

IDs and Dimensions:

idx = blockIdx.x * blockDim.x + threadIdx.x

这里采用SIMD的方式来计算；同一条指令多个线程来执行：

这里一般有四个变量：

blockIdx.x(y,z)
gridDim.x

threadIdx.x(y,z)
blockDim.x

一般写为：
dim3 grid(((nElem-1)/block.x)+1);

同一个block上的thread会执行同样的instruction，这些threads理论上应该是在执行相同的指令；
多个blcoks可以被分配给一个SM，但是不意味着他们会同时执行，取决于可用的资源；

一个kernel对应一个grid（对应整个device），一个grid可以分成多个block；

2. memory Allocation

cudaMalloc(void \*\*  ptr, size_t nbytes )// 这里的size_t 是字节
cudaMemset(void * ptr, int value, size_t count) 
cudaFree(void * ptr)

对应：
malloc()
memset()
free()

eg:

int nbytes = 1024 * sizeof(int)  大小 * sizeof(int)

解释一下cudaMalloc函数：

查看函数说明：

Allocates size bytes of linear memory on the device and returns in *devPtr a pointer to the allocated memory. 

首先要明确这个函数的意义，它指的是在gpu上分配一块指定大小的内存空间，并且让指针指向当前在device上的这块空间，而且这个函数的参数需要传入该指针的原地址：
```
float * dev = Null;
size_t size = 1024 * sizeof(float);
cudaMalloc((void \*\*) &dev, size);
```

cudaMalloc的第一个参数传递的是存储在cpu内存中的指针变量的地址，cudaMalloc在执行完成后，向这个地址中写入了一个地址值（此地址值是GPU显存里的）。这里如果直接穿指针，相当于传入了该指针指向的空间，而不是传入了该指针的地址。

前面的（void\*\*）是强制类型转换；

3. Data copy

cudaMemcpy(void *dst, void *src, size_t nbytes, cudaMemcpyKind direction)

第一个是目的指针
第二个参数是源指针

- 这个函数是线程阻塞，拷贝不完成不会进行下面的其他指令。
- 同样有异步的版本 cudaMemcpyAsync()
- 一个kernel对应一个GRID，该GRID又包含若干个block，block内包含若干个thread。GRID跑在GPU上的时候，可能是独占一个GPU的，也可能是多个kernel并发占用一个GPU的。

Step:

1. identify parallelism
2. write GPU kernel
3. setup the problem
4. Launch the Kernel
5. Copy results back from GPU

## software vs hardware

![compare](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/Toolbox/CUDA/compare.png)

这里的cuda核心叫sp（streaming processor）

　　GPU中每个sm都设计成支持数以百计的线程并行执行，并且每个GPU都包含了很多的SM，所以GPU支持成百上千的线程并行执行。当一个kernel启动后，thread会被分配到这些SM中执行。大量的thread可能会被分配到不同的SM，同一个block中的threads必然在同一个SM中并行（SIMT）执行。每个thread拥有它自己的程序计数器和状态寄存器，并且用该线程自己的数据执行指令，这就是所谓的Single Instruction Multiple Thread。 
  
　　一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。Nvidia把32个threads组成一个warp，warp是调度和运行的基本单元。warp中所有threads并行的执行相同的指令。一个warp需要占用一个SM运行，多个warps需要轮流进入SM。由SM的硬件warp scheduler负责调度。目前每个warp包含32个threads（Nvidia保留修改数量的权利）。所以，一个GPU上resident thread最多只有 SM*warp个。 

    一个SM可以处理的block的数量是根据block之间的内存变量所决定的，看他们有多少个共享变量一个线程使用多少个，一个SM的空间可以提供多少个这样的block共同运行决定的。

## 优化

- global memory

所有线程都可以读取
空间很大，但是访问延迟很高

- shared memory(block中)

data lifetime = block lifetime
block中的线程可以读取自己的shared memory

- registers （thread）

data lifetime = thread lifetime

## Warp

- 可以看成线程执行的队列
- cuda用的是Single Instruction multiple thread，线程将会被分如warp（32个线程），所有在一个warp的线程会在同一时间执行同样的指令；
- 每一个SM会把block分入warps，之后根据可用的硬件资源去调度他们；
- 同一个warp上的thread会有不同的行为？？？
- warp是基本的执行单元，warp内的thread必须执行相同的指令
- 虽然说block、thread都可以是3D，但是在系统看来，都是1D的。
- 最好取的线程数是32的倍数，这样warp不会被浪费
- warp 采用scheduler去调度，每一个SM最多64warps，因为一个cycle中可以处理两个独立的instructions

warps divergence：

<<<<<<< HEAD
## Memory Model

- 我们要尽量做到data locality去最优化memory的实现

- Available Memory 
    - Registers
    - Shared Memory
    - Local Memory
    - Constant Memory
    - Texture Memory
    - Global Memory : in device mem; largest; hign latency; slow; 

### Global Memory:
- 被 half warp / 16个线程同时访问
- 以32，64，128位的片段访问
- 如果可以连续访问比较快，分散访问的化比较慢

所以我们想出的办法是**Memory Coalescing**
1. 尽量一次读取的内存地址是连续的
2. 第一个thread读的地址是64的倍数

这样会提高效率。

### Registers

- fastest
- 在kernel中没有标志符的变量就是有register来存取
- private by thread，数量有限，64 K entries

### shared memory

- fast,low latency, smalled size
- as a block-level
- use __syncthreads() to avoid race conditions
- best way to optimize
- __share__ 开头

the shared memory is divided into banks.


### Texture Memory

### Constant Memory

## 我们关注的高性能计算的两个因素是延迟和吞吐量 latency and bandwidth
=======
同一个warp中的thread，因为分支结构的存在，进入了不同的分支。

bank conflict:

Shared memory 分成16 个 bank。如果同时每个 thread 是存取不同的 bank，就不会产生任何问题，存取 shared memory 的速度和存取寄存器相同。不过，如果同时有两个（或更多个） threads 存取同一个bank 的数据，就会发生 bank conflict，这些 threads 就必须照顺序去存取，而无法同时存取shared memory 了。
>>>>>>> 3ad17c52f6005d5f4701946973d00c05bc95a526
