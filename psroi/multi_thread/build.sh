#!/bin/sh
rm -rf multithreading_psroi single_psroi
gcc -c parallel.c -w -O3 -o parallel.o
gcc -c psroi_pooling.c -w -O3 -o psroi_pooling.o
gcc parallel.o psroi_pooling.o -lm -lpthread -o  multithreading_psroi
rm -rf *.o

#-ffast-math配合O2以上的gcc优化：ceil函数出现问题；
#在一个接近float32取值精度极限的一个表达式放入ceil中导致向上取整多1
#这里面是含有变量的表达式，如果是特定的数字组成的计算式放入ceil中则没有问题
#gcc single_psroi.c -w -O3 -ffast-math -lm -o single_psroi

gcc single_psroi.c -w -O3 -lm -o single_psroi


./single_psroi
diff ctop.bin psroipooled_cls_rois.bin

./multithreading_psroi
diff ctop.bin psroipooled_cls_rois.bin
