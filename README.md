# rcnn-optimize
## rfcn/faster rcnn optimize
### 重写计算库/并行加速/模块C改写和算法改进/网络模型的裁剪/fpga驱动
* rpn: C code
* roi: multi-thread C code
* Vector * Vector :avx test
* level\_thread: multi-thread matrix multiple sgemm  demo form openblas

* ristretto_rfcn: R-FCN 自动定点化工具实现，将ristretto移植进入faster rcnn框架下
* A10_FPGA_Dma: Altera 10 FPGA DMA驱动
* rfcn_net: rfcn非卷积部分实现(softmax rpn psroi ave_pooling)以及最后的box regress, nms处理
* rpn_optimize: 对rpn层的优化,算法改进等处理，性能提高20~100倍
* net_modle_optimize: 对基础ZF Net网络模型的优化,借鉴Inception、shuffle net、Crelu、dw conv等等结构，性能提升2~8倍
### 图像视频处理：OpenCV GPU/GPU NPP/Ffmpeg Intel硬件解码、IPP图片转换
