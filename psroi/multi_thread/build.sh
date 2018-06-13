#!/bin/sh
rm -rf multithreading_psroi single_psroi
gcc -c parallel.c -w -O3 -o parallel.o
gcc -c psroi_pooling.c -w -O3 -o psroi_pooling.o
gcc parallel.o psroi_pooling.o -lm -lpthread -o  multithreading_psroi
rm -rf *.o

gcc single_psroi.c -w -O3 -lm -o single_psroi


./single_psroi
diff ctop.bin psroipooled_cls_rois.bin

./multithreading_psroi
diff ctop.bin psroipooled_cls_rois.bin
