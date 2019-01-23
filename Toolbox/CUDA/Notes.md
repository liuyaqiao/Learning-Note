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
    - CUDA cores
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

多个SP加上其他的一些资源组成一个streaming multiprocessor。也叫GPU大核，其他资源如：warp scheduler，register，shared memory等。SM可以看做GPU的心脏（对比CPU核心），register和shared memory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。

在kepler中，一个逻辑控制单元控制32个cores，避免了因为逻辑控制单元太少而造成的cuda核心的冗余。

## Memory Hierarchy:

global mem ： 访问的缓存延迟很高，所以使用了L2

warp -- 通过切换来隐藏延迟

## CUDA Program

1. kernel 函数

thread :
- all threada execute same sequential program
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

同一个block上的thread会执行同样的instruction，这些threads理论上应该是在执行相同的指令；
多个blcoks可以被分配给一个SM，但是不意味着他们会同时执行，取决于可用的资源；
一个kernel对应一个grid（对应整个device），一个grid可以分成多个block（一个block对应了一个SM），

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

- cuda用的是Single Instruction multiple thread，线程将会被分如warp（32个线程），所有在一个warp的线程会在同一时间执行同样的指令；
- 每一个SM会把block分入warps，之后根据可用的硬件资源去调度他们；
- 同一个warp上的thread会有不同的行为？？？
- warp是基本的执行单元，warp内的thread必须执行相同的指令
- 虽然说block、thread都可以是3D，但是在系统看来，都是1D的。
- 最好取的线程数是32的倍数，这样warp不会被浪费
- warp 采用scheduler去调度，每一个SM最多64warps，因为一个cycle中可以处理两个独立的instructions

warps divergence：