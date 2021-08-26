====================1.使用model-profiler产生量化表; 2.执行默认转换;====================================
../glow/build_Debug/bin/model-profiler -model=facerecong.onnx -dump-profile=profile.yaml \
-input-dataset=input.1,rawbin,dir,data/facerecong_data/gray_quan_img_500_bin

../glow/build_Debug/bin/model-compiler -load-profile=profile.yaml -model=./facerecong.onnx \
 -model-input=input.1,float,[1,1,56,56]  -backend=CPU -target=x86_64  -emit-bundle=facerecong_x86_quanta \
-keep-original-precision-for-nodes=Div 

../glow/build_Debug/bin/model-profiler -model=live_128.onnx -dump-profile=profile.yaml \
-input-dataset=input,rawbin,dir,data/live_data/gray_quan_128/

../glow/build_Debug/bin/model-compiler -load-profile=profile.yaml -model=./live_128.onnx \
 -model-input=input,float,[1,1,1,128]  -backend=CPU -target=x86_64  -emit-bundle=live_128_x86_quanta \
-keep-original-precision-for-nodes=Div

../glow/build_Debug/bin/model-profiler -model=lt_floor.onnx -dump-profile=profile.yaml \
-input-dataset=input_data,rawbin,dir,data/lm_data/gray_quan_img/

../glow/build_Debug/bin/model-compiler -load-profile=profile.yaml -model=./lt_floor.onnx  \
-backend=CPU -target=x86_64  -emit-bundle=lt_floor_x86_quanta \
-keep-original-precision-for-nodes=Div 
==========================================GLOW编译器编译ONNX模型=====================================================
https://github.com/pytorch/glow/blob/master/docs/AOT.md
https://github.com/pytorch/glow
