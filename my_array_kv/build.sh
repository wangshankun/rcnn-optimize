#!/bin/sh

rm -rf test test.db
g++ -std=c++11 main.cpp SimpleHash.cpp CityHash64.cpp -O3 -o test
./test


