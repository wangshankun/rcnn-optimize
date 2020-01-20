#!/bin/sh

PWD=$(pwd)
rm -rf ./target
#RUSTFLAGS="-C link-args=-Wl,-rpath,${PWD}/../jpg_2_h265/:${PWD}/../h265_2_jpg/" cargo +nightly  build --release -vv
RUSTFLAGS="-C link-args=-Wl,-rpath,${PWD}/../jpg_2_h265/:${PWD}/../h265_2_jpg/" cargo +nightly  build --release

#gcc -O3 -w call_rust.c -o call_rust -L./target/release -lrustcalls -Wl,-R./target/release

#./call_rust
