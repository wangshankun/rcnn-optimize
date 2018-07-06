*         **altera 10 fpga dma**

* altera 10 fpga dma 驱动和应用层读写接口的整体实现
* 采用User-Space Device Drivers的想法：
*     驱动内核层仅仅做资源申请，不做逻辑；
*     驱动应用层实现所有驱动的配置和逻辑处理；
*  应用层通过调用mmap方法，其在内核层中对应调用mmap_device_mem，
*  把内核层已经和硬件绑定好的物理空间等资源映射给应用层，这样应用层
*  可以直接配置设备的寄存器。
* 
*  参考代码G2x4_avmm_dma_Linux.tar.gz

