# openblas multithreading frameworks 
## pull the api sgemm from openblas code

# 从openblas源码中抽出的其demo骨架，例子是sgemm api实现

![image](https://github.com/wangshankun/rcnn-optimize/blob/master/level_thread/readme.jpg)

   王茜openblas早期作者之一，在她的博士论文中《多核CPU上稠密线性代数函数优化及自动代码生成研究》
   主要描述就是openblas框架作为一个软件工程具备优点，很有借鉴意义。
   
   而openblas中对cache和tlb的优化完全来自于gotoblas,这部分可以参考国内《GOTOBLAS一般矩阵乘法高效实现机制的研究》
   
   或者更早的一篇文章《On Reducing TLB Misses in Matrix Multiplication》

   本人借鉴其思想实现了一个全连接计算的demo
   https://github.com/wangshankun/rcnn-optimize/tree/master/pthread_fc_cache_and_sharejob
   
   以及，采用其多线程的架构，实现了caffe layer的加速
   https://github.com/wangshankun/multithreading_frcnn
   
