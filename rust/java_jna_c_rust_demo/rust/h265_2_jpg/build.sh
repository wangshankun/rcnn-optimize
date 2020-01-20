#!/bin/sh

nvcode_path=Video_Codec_SDK_9.0.20/
cuda_path=/usr/local/cuda-10.1/

g++ -std=c++11 -O3 -fPIC -o NvDecoder.o -c ${nvcode_path}/Samples/NvCodec/NvDecoder/NvDecoder.cpp -I/usr/local/cuda-10.1/include -I ${nvcode_path}/Samples/NvCodec/NvDecoder/ -I ${nvcode_path}/include/ -I Video_Codec_SDK_9.0.20/Samples/Utils/ -I ${nvcode_path}/Samples/NvCodec/

#g++ -std=c++11 -c AppDec.cpp -o AppDec.o -I/usr/local/cuda-10.1/include -I ${nvcode_path}/Samples/NvCodec/NvDecoder/ -I ${nvcode_path}/include/ -I Video_Codec_SDK_9.0.20/Samples/Utils/

#g++ -std=c++11 -o AppDec AppDec.o NvDecoder.o  -L /usr/local/lib -L ${cuda_path}/lib64 -L ${cuda_path}/lib64/stubs -ldl -lcuda -lnvcuvid -lavcodec -lavutil -lavformat -lnppisu_static -lnpps_static -lnppial_static -lnppist_static -lnppidei_static -lnppif_static -lnppim_static -lnppig_static -lnppicc_static -lnppicom_static -lnppitc_static -lnppc_static  -lculibos -lcudart_static -lpthread -lrt -lnvjpeg


g++ -std=c++11 -O3 -fPIC -c decompress_lib.cpp -o decompress_lib.o -I/usr/local/cuda-10.1/include -I ${nvcode_path}/Samples/NvCodec/NvDecoder/ -I ${nvcode_path}/include/ -I Video_Codec_SDK_9.0.20/Samples/Utils/

gcc -O3 -m64 *.o  -o libdecompress.so -shared -L /usr/local/lib -L ${cuda_path}/lib64 -L ${cuda_path}/lib64/stubs -ldl -lcuda -lnvcuvid -lavcodec -lavutil -lavformat -lnppisu_static -lnpps_static -lnppial_static -lnppist_static -lnppidei_static -lnppif_static -lnppim_static -lnppig_static -lnppicc_static -lnppicom_static -lnppitc_static -lnppc_static  -lculibos -lcudart_static -lpthread -lrt -lnvjpeg

gcc -O3 -m64 main.c -o main -ldecompress -L ./ -Wl,-rpath,./

rm -rf *.o
