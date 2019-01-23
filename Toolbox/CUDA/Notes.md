# GPU Programming

## CUDA API

1. two level
2. 功能

## CUDA concept：

1. host device kernel complier
2. 执行cuda程序的过程
3. cuda中grid、block、thread结构

## GPU Architecture：

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


Step:

1. identify parallelism
2. write GPU kernel
3. setup the problem
4. Launch the Kernel
5. Copy results back from GPU

## 优化

- global memory

所有线程都可以读取
空间很大，但是访问延迟很高

- shared memory(block中)

data lifetime = block lifetime
block中的线程可以读取自己的shared memory

- registers （thread）

data lifetime = thread lifetime