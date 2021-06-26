#!/bin/sh

WSPACE=$(pwd)
rm -rf build
mkdir build
cd build

g++ -c -I"$WSPACE/model_bin" $WSPACE/src/sail_face_watch.cpp -o sail_face_watch.o 
ar -cr libsail_face_watch.a sail_face_watch.o  \
      $WSPACE/model_bin/live_128.o $WSPACE/model_bin/lt_floor.o $WSPACE/model_bin/facerecong.o 
g++ -no-pie $WSPACE/example/main.cpp  -L./ -lsail_face_watch -lm -o main
rm -rf *.o 
./main $WSPACE/example/1.bin 
