# CUSR -- A Genetic Programming Based Symbolic Regression Framework Implemented by CUDA C/C++



## 介绍

这项工作利用GPU加速基于GP的符号回归。这个框架的参数和训练方式采用了sklearn风格。



## 框架参数

| 参数                     | 类型               | 解释                                                         |
| ------------------------ | ------------------ | ------------------------------------------------------------ |
| population_size          | int                | --                                                           |
| generations              | int                | --                                                           |
| tournament_size          | int                | --                                                           |
| stopping_criteria        | float              | 如果 metric <= stopping_criteria那么停止迭代                 |
| constant_range           | pair<float, float> | termianl节点常量值范围为[constant_range.first, constant_range.second] |
| init_depth               | pair<int, int>     | 初始化表达式树的深度范围为[init_depth.first, init_depth.second] |
| init_method              | InitMemthod        | 种群初始化的方法                                             |
| function_set             | vector<Function>   | 表达式树中使用的函数集合                                     |
| metric                   | Metric             | 损失类型，提供MSE、MAE、RMSE                                 |
| restrict_depth           | bool               | 是否限制program的深度                                        |
| max_program_depth        | int                | 表达式树的最大深度，基于GP的符号回归一般这个数量小于17       |
| parsimony_coefficient    | float              | 对树的节点数量的惩罚，因为我们希望得到的表达式尽可能的简洁。在选择时 loss' = loss + program.length * parsimony_coefficient |
| p_crossover              | float              | --                                                           |
| p_subtree_mutation       | float              | --                                                           |
| p_hoist_mutation         | float              | --                                                           |
| p_point_mutation         | float              | --                                                           |
| p_point_replace          | float              | --                                                           |
| p_constant               | float              | terminal生成常量的概率                                       |
| use_gpu                  | bool               | 是否使用GPU加速                                              |
| best_program             | Program            | 记录当前种群中损失最小的个体                                 |
| best_program_in_each_gen | vector<Program>    | 记录每代种群中损失最小的个体                                 |
| regress_time_in_sec      | float              | 记录本次回归任务中所花费的时间                               |



## 使用这个框架

#### 系统环境

CUDA Version >= 9.0



#### 自定义参数

```c++
/* main.cu */

vector<vector<float>> dataset; // 数据集
vector<float> real_value; // 真实值

/******************************************************************************************
 ** 框架需要的 dataset 和 real_value 的格式
 ** 例如我们希望框架回归 f(a, b) = a + b，我们可以准备如下数据：（示例中不考虑采样的范围和分布）
	dataset = {
		{1, 2},
		{2, 3},
		{1, 1},
		{....}
	};
	
	real_value = {
		3,
		5,
		2,
		...
	};
	
	** 同理，如果需要回归 f(a, b, c) = a + b + c，可以准备如下数据：（示例中不考虑采样的范围和分布）
	dataset = {
		{1, 2, 3},
		{2, 3, 4},
		{3, 4, 5},
		{.......}
	};
	
	real_value = {
		6,
		9,
		12,
		...
	};
	
********************************************************************************************/


/* 如果有多个GPU，选择指定的GPU */
void choose_gpu() {
	int count;
  cudaGetDeviceCount(&count);
  
  if(count == 0) {
    fprintf(stderr, "There is no device./n");
    return;
  }
  
  cout << "input GPU id" << endl;
  int gpu_id;
  cin >> gpu_id;
  cudaSetDevice(gpu_id);
}


int main() {
	 // 如果有多个GPU，选择指定的GPU，如果只有一个GPU，那么可以不要这一步
   choose_gpu();
  
   // 生成 dataset 和 real_value，这个请自行实现
   gen_dataset();
   
   // 创建一个做符号回归的Engine
   cusr::RegressionEngine reg;
  
   // 设置各种参数
   reg.function_set = { _add, _cos, _sub, _div, _tan, _mul, _sin };
   reg.use_gpu = true;            
   reg.max_program_depth = 10;                      
   reg.population_size = 50;
   reg.generations = 50;
   reg.parsimony_coefficient = 0;        
   reg.const_range = {-5, 5};     
   reg.init_depth = {4, 10};      
   reg.init_method = init_t::half_and_half;  
   reg.tournament_size = 3;                  
   reg.metric = metric_t::root_mean_square_error; 
  
   // 进行训练
   reg.fit(dataset, real_value); 

   // 训练后得到结果
   cout << "Execution Time: " << reg.regress_time_in_sec << endl;
   cout << "Best Fitness  : " << reg.best_program.fitness << endl;
  
   // 打印表达式树的前缀表达式和中缀表达式
   cout << "Best Program (in prefix):  " << cusr::program::prefix_to_string(reg.best_program.prefix) << endl;
   cout << "Best Program (in infix) :  " << cusr::program::prefix_to_infix(reg.best_program.prefix) << endl;
   return 0;
}
```



#### 编译

```shell
nvcc -o sr main.cu src/*.cu
```

#### 运行

```she
./sr
```



## 文件结构

project文件结构：

```
|--cusr
  |--src
    |--prefix.cuh     // 一些包含前缀表达式的操作。包括生成随机的表达式树，将树转换成前缀表达式等
    |--program.cuh    // 在program层面的操作，比如crossver、mutation等操作
    |--fit_eval.cuh   // 使用GPU进行适应度评估的工作
    |--regression.cuh // 对回归任务所需的各类功能封装成框架，具有sklearn风格
    |--prefix.cu      // prefix.cuh实现
    |--program.cu     // program.cuh实现
    |--fit_eval.cu    // fit_eval.cuh实现
    |--regression.cu  // regression.cuh实现
  |--include
    |--cusr.h         // 包含该项目时仅需引用该头文件 #include "include/cusr.h"
```



## 核心算法及实现细节

这项工作加速了GP的适应度评估阶段，从而实现对符号回归的加速。回归过程的流程图如下：

![cuda_process](./pic/cuda_process.png)



为了做到最大程度利用GPU性能，我们希望GPU端算法实现合并访存，且避免warp divergence。为此，我们设计了如下的设备端数据结构和响应加速算法：

1. 通过列优先的方式存储数据集。如图：列优先的存储可以保证线程在访存时访问全局内存的相邻地址，这样cache会将线程的访存请求合并，从而得到加速。![column_major](./pic/column_major.png)

   

2. 将表达式树使用前缀表达式表示，并存储在常量内存中。得益于广播机制，常量warp中的线程访问常量内存中的内容仅需一次访存周期。为了确定节点的类型和它的值从而执行不同的操作，我们用两个值来唯一确定一个节点。

   ![cuda_copy](./pic/cuda_copy.png)

实现常量内存拷贝的代码如下：

```c++
/*************************************************************
这部分的代码位于fit_eval.cu calSingleProgram() 函数中
**************************************************************/

// 前缀表达式长度的最大值，也因此需要限制树的深度，否则可能会造成溢出
#define MAX_PREFIX_LEN 2048 

// 声明两个数组，分别存储节点的类型和值
__constant__ float d_nodeValue[MAX_PREFIX_LEN];
__constant__ float d_nodeType[MAX_PREFIX_LEN];

void copyProgram() {
  // 主机端的两个数组
	float h_nodeValue[MAX_PREFIX_LEN];
	float h_nodeType[MAX_PREFIX_LEN];
  
  // 将program拆分为两个数组
	for (int i = 0; i < program.length; i++) {
  	char type = program.prefix[i].node_type;
    h_nodeType[i] = type;
      if (type == 'c') {
        h_nodeValue[i] = program.prefix[i].constant;
      } else if (type == 'v') {
        h_nodeValue[i] = program.prefix[i].variable;
      } else // if (type == 'u' || type == 'v')
      {
        h_nodeValue[i] = program.prefix[i].function;
      }
  }
  
  // 拷贝：主存-->GPU常量内存
  cudaMemcpyToSymbol(d_nodeValue, h_nodeValue, sizeof(float) * program.length);
  cudaMemcpyToSymbol(d_nodeType, h_nodeType, sizeof(float) * program.length);
}
```



3. GPU端栈。进行前缀表达式的求值需要栈这种数据结构。我们的算法让每个线程计算program在一个数据点下的损失。在GPU端为了保证实现合并访存，我们的栈结构保证在进行栈操作时能够合并访存。注意⚠️这种满足合并访存的栈结构带来的性能上的提升是决定性的。我们的栈结构如图：

   ![stack](./pic/stack.png)

我们的栈本质上是一个一维数组。如图所示：当top=0时所有线程访问左侧虚线框中的元素，当top=1时访问右侧虚线框中的元素。我们为每个block的每个线程开辟独立的放置元素的空间（在我们的实现中我们规定每个block使用512个线程）。那么开辟多大空间合适呢？事实上top的最大值是max_program_depth + 1就足够了。在我的实现中我们没有封装单独的栈结构类，当然也可以对这个数据结构 class DeviceStack 进行封装，并定义\_\_device\_\_ void push() 和 \_\_device\_\_ float pop() 等。栈空间的开辟代码如下：

```c++
/*************************************************************
这部分的代码位于fit_eval.cu calSingleProgram() 函数中
**************************************************************/

float *mallocStack(int blockNum) {
	float *stack;
	// allocate stack space, the size of which = sizeof(float) * THREAD_PER_BLOCK * (maxDepth + 1)
	cudaMalloc((void **) &stack, sizeof(float) * THREAD_PER_BLOCK * (DEPTH + 1) * blockNum);
	return stack;
}
```

已知栈顶top，每个线程此时应当访问的内存地址呢？

```c++
float element = stack[THREAD_PER_BLOCK * (DEPTH + 1) * blockIdx.x + top * THREAD_PER_BLOCK + threadIdx.x];
```

在完成表达式树的计算时，计算结果位于top=0处（也就是上图左侧虚线框）。



4. 并行规约。并行规约将计算出的所有损失并行求和。损失的计算结果会从栈顶取出，然后放到共享内存中。共享内存只能被同一个block中的线程访问，但是它访存速度很快，非常适合做规约。实现时我们避免了bank conflict问题，保证了效率。

![reduction](./pic/reduction.png)

实现代码如下（这部分代码在kernel中）：

```c++
if (threadIdx.x < 256) { shared[threadIdx.x] += shared[threadIdx.x + 256]; }
 __syncthreads();

if (threadIdx.x < 128) { shared[threadIdx.x] += shared[threadIdx.x + 128]; }
__syncthreads();

if (threadIdx.x < 64) { shared[threadIdx.x] += shared[threadIdx.x + 64]; }
__syncthreads();

// 下面不需要 __syncthreads()了，因为一个warp中的32个线程是GPU的最小执行单元，因此它们从来都是同步的
if (threadIdx.x < 32) { shared[threadIdx.x] += shared[threadIdx.x + 32]; }
if (threadIdx.x < 16) { shared[threadIdx.x] += shared[threadIdx.x + 16]; }
if (threadIdx.x < 8) { shared[threadIdx.x] += shared[threadIdx.x + 8]; }
if (threadIdx.x < 4) { shared[threadIdx.x] += shared[threadIdx.x + 4]; }
if (threadIdx.x < 2) { shared[threadIdx.x] += shared[threadIdx.x + 2]; }
if (threadIdx.x < 1) {
  shared[threadIdx.x] += shared[threadIdx.x + 1];
  // 将这512个结果存储到 result[] 中，result[] 中的结果的求和由CPU串型执行就够了
  result[blockIdx.x] = shared[0];
}
```



总结一下设备端的数据流就像这样：

![algo_mem](./pic/algo_mem.png)

核函数：

```c++
/*************************************************************
这部分的代码位于fit_eval.cu calSingleProgram() 函数中
**************************************************************/

// stack offset，这个宏定义使得下面的代码在访问栈时不必写这么长的下标了，仅需：stack[S_OFF]就可以了
#define S_OFF THREAD_PER_BLOCK * (DEPTH + 1) * blockIdx.x + top * THREAD_PER_BLOCK + threadIdx.x

__global__ void
calFitnessGPU(int len, float *ds, int dsPitch, float *label, float *stack, float *result, int dataset_size) {
  // 获得共享内存空间，并将里面的值填充全0
  extern __shared__ float shared[];
  shared[threadIdx.x] = 0;
  
  // 当前线程负责的数据集的下标
  int dataset_no = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;
  
  // 由于每个block固定使用512个线程，因此当前线程可能会超过数据集的大小
  if (dataset_no < dataset_size) {
    int top = 0;
    
    // 从后往前遍历每个节点
    for (int i = len - 1; i >= 0; i--) {
      int node_type = d_nodeType[i];
      float node_value = d_nodeValue[i];

      // 对于不同的节点类型，我们做对应的栈操作
      if (node_type == CONSTANT) {
        stack[S_OFF] = node_value;
        top++;
      } else if (node_type == VARIABLE) {
        int var_num = node_value;
        stack[S_OFF] = ((float *) ((char *) ds + var_num * dsPitch))[dataset_no];
        top++;
      } else if (node_type == UNARY_FUNCTION) {
        int function = node_value;
        top--;
        float var1 = stack[S_OFF];
        if (function == SIN_SIGN) {
          stack[S_OFF] = std::sin(var1);
          top++;
        } else if (function == COS_SIGN) {
          stack[S_OFF] = std::cos(var1);
          top++;
        } else if (function == TAN_SIGN) {
          stack[S_OFF] = std::tan(var1);
          top++;
        } else if (function == LOG_SIGN) {
          if (var1 <= 0) {
            stack[S_OFF] = -1.0f;
            top++;
          } else {
            stack[S_OFF] = std::log(var1);
            top++;
          }
        } else if (function == INV_SIGN) {
          if (var1 == 0) {
            var1 = DELTA;
          }
          stack[S_OFF] = 1.0f / var1;
          top++;
        }
      } else { // (node_type == BINARY_FUNCTION)
        int function = node_value;
        top--;
        float var1 = stack[S_OFF];
        top--;
        float var2 = stack[S_OFF];
        if (function == ADD_SIGN) {
          stack[S_OFF] = var1 + var2;
          top++;
        } else if (function == SUB_SIGN) {
          stack[S_OFF] = var1 - var2;
          top++;
        } else if (function == MUL_SIGN) {
          stack[S_OFF] = var1 * var2;
          top++;
        } else if (function == DIV_SIGN) {
          if (var2 == 0) {
            var2 = DELTA;
          }
          stack[S_OFF] = var1 / var2;
          top++;
        } else if (function == MAX_SIGN) {
          stack[S_OFF] = var1 >= var2 ? var1 : var2;
          top++;
        } else if (function == MIN_SIGN) {
          stack[S_OFF] = var1 <= var2 ? var1 : var2;
          top++;
        }
      }
    }
    
    // 计算损失
    top--;
    float prefix_value = stack[S_OFF];
    float label_value = label[dataset_no];
    float loss = prefix_value - label_value;
    float fitness = loss * loss;
    
    // 将损失存储到共享内存
    shared[threadIdx.x] = fitness;
  }
  
  // 线程同步，block中的所有线程需要在这个点等待直到block中所有的线程都执行到这里才会往下执行
  __syncthreads();
  
  // 并行规约
#if THREAD_PER_BLOCK >= 1024
  if (threadIdx.x < 512) { shared[threadIdx.x] += shared[threadIdx.x + 512]; }
  __syncthreads();
#endif
#if THREAD_PER_BLOCK >= 512
  if (threadIdx.x < 256) { shared[threadIdx.x] += shared[threadIdx.x + 256]; }
  __syncthreads();
#endif
  if (threadIdx.x < 128) { shared[threadIdx.x] += shared[threadIdx.x + 128]; }
  __syncthreads();
  if (threadIdx.x < 64) { shared[threadIdx.x] += shared[threadIdx.x + 64]; }
  __syncthreads();
  if (threadIdx.x < 32) { shared[threadIdx.x] += shared[threadIdx.x + 32]; }
  if (threadIdx.x < 16) { shared[threadIdx.x] += shared[threadIdx.x + 16]; }
  if (threadIdx.x < 8) { shared[threadIdx.x] += shared[threadIdx.x + 8]; }
  if (threadIdx.x < 4) { shared[threadIdx.x] += shared[threadIdx.x + 4]; }
  if (threadIdx.x < 2) { shared[threadIdx.x] += shared[threadIdx.x + 2]; }
  if (threadIdx.x < 1) {
    shared[threadIdx.x] += shared[threadIdx.x + 1];
    // 将这512个损失的和存放于result[]中
    result[blockIdx.x] = shared[0];
  }
}
```



