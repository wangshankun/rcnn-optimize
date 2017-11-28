#!/bin/sh
rm -rf rpn_proposal
gcc rpn_proposal.c -w -lm -ffast-math -O3 -o rpn_proposal
