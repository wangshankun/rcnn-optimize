#!/bin/sh

g++ -std=c++11 main.cpp http_client.cpp -lrt -lpthread -lcrypto -ljsoncpp -lcurl -o client -Wl,-rpath,/usr/local/lib64/ 


