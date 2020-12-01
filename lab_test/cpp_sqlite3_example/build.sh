#!/bin/sh

gcc main.c -lsqlite3 -o test && ./test
