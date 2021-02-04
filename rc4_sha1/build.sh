
#!/bin/sh
g++ -std=c++11 ARC4.cpp testRC4.cpp -I ./ -o rc4

gcc testSHA.c -I ./ -o sha1
