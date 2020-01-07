#!/bin/sh
RUST_FUN_PATH=$(pwd)/rust/rustcalls/target/release/
cd $(pwd)/rust/rustcalls && ./build.sh
cd -
gcc -w -fpic -c SHIC.c
gcc -shared -o libSHIClib.so SHIC.o -L${RUST_FUN_PATH} -lrustcalls -Wl,-R${RUST_FUN_PATH}
export CLASSPATH=$CLASSPATP:./jna.jar
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./

javac Main.java

java Main

