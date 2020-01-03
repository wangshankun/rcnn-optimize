#!/bin/sh

cargo +nightly build

gcc -g -w call_rust.c -o call_rust -L./target/debug -lrustcalls -Wl,-R./target/debug

./call_rust
