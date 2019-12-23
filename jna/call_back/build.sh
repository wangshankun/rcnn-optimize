#!/bin/sh

export CLASSPATH=$CLASSPATP:./jna.jar
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./

gcc -fpic -c test.c

gcc -shared -o libtestlib.so test.o

javac TestSo.java

java TestSo
