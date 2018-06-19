# C code for fastercnn rpn proposal layers and optimize
* **change the order sort and filter, out of the cycle earlier**

* **don't need reshape**
*               **rk3399 A72 Test**
* faster rcnn      rpn elapsed time:0.192554(300 prob)
* this             rpn elapsed time:0.023223(300 prob)
* using threshold  rpn elapsed time:0.010289(45 prob)
