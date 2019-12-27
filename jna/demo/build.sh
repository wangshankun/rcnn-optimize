#!/bin/sh
gcc -w -fpic -c SHIC.c
gcc -shared -o libSHIClib.so SHIC.o
export CLASSPATH=$CLASSPATP:./jna.jar
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./

javac Main.java

java Main

