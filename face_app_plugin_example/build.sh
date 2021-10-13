#!/bin/sh

WSPACE=$(pwd)
rm -rf $WSPACE/build
mkdir $WSPACE/build
cd $WSPACE/build

g++ -g -O0   \
    $WSPACE/src/sail_face.cpp  \
    $WSPACE/src/generate_face_id/generate_face_id.cpp \
    $WSPACE/src/face_db/hashmap_db.cpp \
    $WSPACE/src/face_db/city_hash64.cpp \
    -I$WSPACE/src/generate_face_id/deep_learning_model_exe/ \
    -I$WSPACE/src/generate_face_id/ \
    -I$WSPACE/src/face_db/ \
    -ldl \
    -fPIC -shared  -o libsail_face.so

g++  -g -O0 \
     $WSPACE/example/main.cpp  \
     -L./  -lsail_face -Wl,-rpath . \
     -o test

./test
