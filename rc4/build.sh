#!/bin/sh
g++ test.cpp -lssl -lcrypto -o test

./test

diff 1.jpg 1.jpg.idst_denc

diff 1.jpg 1.jpg.denc


hexdump  -C 1.jpg.idst_enc|more

