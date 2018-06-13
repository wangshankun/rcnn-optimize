#!/bin/sh
#gcc psroi.c -w -O3 -ffast-math -lm -o psroi
#./psroi

gcc psroi.c -w -O3 -lm -o psroi
./psroi
diff ctop.bin psroipooled_cls_rois.bin
