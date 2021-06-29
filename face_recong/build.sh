#!/bin/sh
gcc_version=`g++ -dumpversion`
gcc_maj_v=`echo ${gcc_version} | cut -c 1-1`

if [ "$gcc_maj_v" -lt 6 ]
then
     opt=""
else
     opt="-no-pie"
fi


WSPACE=$(pwd)
rm -rf $WSPACE/build
mkdir $WSPACE/build
cd $WSPACE/build

#编译人脸库
g++ $opt -std=c++11 -g -c -I"$WSPACE/model_bin" $WSPACE/src/sail_face_watch.cpp -o sail_face_watch.o 
ar -cr libsail_face_watch.a sail_face_watch.o  \
      $WSPACE/model_bin/live_128.o $WSPACE/model_bin/lt_floor.o $WSPACE/model_bin/facerecong.o 

#编译jpg解码
cd $WSPACE/example/jpg_decoder && make

#链接
cd $WSPACE/build
g++ $opt -std=c++11 -g  $WSPACE/example/main.cpp  \
     -L"$WSPACE/build" -lsail_face_watch -lm \
     -L"$WSPACE/example/jpg_decoder" -lffjpeg \
     -I"$WSPACE/example/jpg_decoder" \
     -o main

#清理编译中间结果
rm -rf *.o 
cd $WSPACE/example/jpg_decoder && make clean

#执行测试程序
cd $WSPACE/build
export WATCH_TEST_PATH="$WSPACE/example/test_data"
./main
