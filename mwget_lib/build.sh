#!/bin/sh

./configure 

make -j$(nproc)

cd src/

g++ -fPIC -DHAVE_CONFIG_H -I. -I..  -DLOCALEDIR=\"/usr/local/share/locale\" -c mwget_entrance.cpp -o mwget_entrance.o

rm -rf mwget.o 

g++ *.o  -o libmwget.so -shared  -lssl -lcrypto -lpthread

g++ test.cpp -o test -lmwget -L./

