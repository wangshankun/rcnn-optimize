
## openblas看似很大其实只是因为需要实现众多API,而绝大多数情况下我们只需要用sgemm这一个API,因此就做了这个尝试,从openblas中抽出的sgemm实例,并且保留了openblas的多线程和内存优化的部分,最终只需几个文件可以达到一样的效果
## openblas multithreading frameworks; pull the api sgemm from openblas code

![image](https://github.com/wangshankun/rcnn-optimize/blob/master/level_thread/readme.jpg)

  openblas加速原理参考文章:
     《GOTOBLAS一般矩阵乘法高效实现机制的研究》
     《On Reducing TLB Misses in Matrix Multiplication》
  矩阵乘优化的一步步实现,参考代码:
      https://github.com/wangshankun/rcnn-optimize/tree/master/optimize_gemm_step_by_step
      
  
   
