# C code for fastercnn rpn proposal layers and optimize

* **安照caffe卷积出来的格式进入rpn layer**
* scores shape (1, 42, 15, 26)
* bbox deltas shape (1, 84, 15, 26)
* base anchor is faster rcnn format(from file generate\_anchors.py)

* **1.按照score筛掉不合格(比如低于0.5)anchor,因此绝大多数anchor都是没什么得分,因此这步可以筛下不少计算量**
* **2.不再根据feature map生成所有anchor，而是根据合格得分anchor的index参照base anchor生成对应点的几个anchor**
* **3.不论是box filter循环还是nms循环，完成指定数目主动跳出循环而不是继续执行到底**
* **4.如果anchor数量过多，qsor排序会成瓶颈，可以加入多线程加快排序**

*               **rk3399 A72 Test**
* using 0.5 score threshold  rpn elapsed time:0.00079638(99 prob)
* using 0.6 score threshold  rpn elapsed time:0.00009081(8 prob)
