#!/bin/sh

#g++ -DSQLITE_HAS_CODEC main.cpp -lsqlite3 -o test && ./test
g++  main.cpp -lsqlite3 -o test && ./test
