#!/bin/sh

cargo +nightly build --release

#gcc -O3 -w call_rust.c -o call_rust -L./target/debug -lrustcalls -Wl,-R./target/debug
gcc -O3 -w call_rust.c -o call_rust -L./target/release -lrustcalls -Wl,-R./target/release

./call_rust
