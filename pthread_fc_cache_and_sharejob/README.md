# 采用openblas架构和原理实现的一个全连接层计算例子
* 使用C内嵌intel avx实现的计算核 ，虽然降低些速度，但使得更方便移植和改写
* 使用openblas的share job思想
* 使用gotoblas对cache控制的思想

## 8核 CPU 可以达到6~7倍提速与原生openblas速度几乎无差异

    perf stat ./fc_pthread_nt 
    A read size:2764800 
    B read size:37748736  
    mypos:3 elapsed1:0.000724 elapsed2:0.002428 elapsed3:0.012894 elapsed4:0.086158
    mypos:4 elapsed1:0.000729 elapsed2:0.002314 elapsed3:0.012592 elapsed4:0.086655
    mypos:5 elapsed1:0.000967 elapsed2:0.002382 elapsed3:0.012433 elapsed4:0.086564
    mypos:6 elapsed1:0.001530 elapsed2:0.002252 elapsed3:0.012294 elapsed4:0.086434
    mypos:1 elapsed1:0.000723 elapsed2:0.002330 elapsed3:0.012760 elapsed4:0.088897
    mypos:2 elapsed1:0.000745 elapsed2:0.002385 elapsed3:0.012800 elapsed4:0.088796
    mypos:7 elapsed1:0.000437 elapsed2:0.002353 elapsed3:0.012527 elapsed4:0.086813
    mypos:0 elapsed1:0.000991 elapsed2:0.002399 elapsed3:0.012773 elapsed4:0.088968
    ww_sgemm_thread_nt elapsed time:0.117301 cbuff[3580602]:0.001318
    failed to read counter stalled-cycles-frontend
    failed to read counter stalled-cycles-backend

     Performance counter stats for './fc_pthread_nt':

            935.227243      task-clock (msec)         #    6.556 CPUs utilized          
                 3,179      context-switches          #    0.003 M/sec                  
                    13      cpu-migrations            #    0.014 K/sec                  
                 2,661      page-faults               #    0.003 M/sec                  
         3,457,154,668      cycles                    #    3.697 GHz                    
       <not supported>      stalled-cycles-frontend  
       <not supported>      stalled-cycles-backend   
         5,985,734,188      instructions              #    1.73  insns per cycle        
           298,273,231      branches                  #  318.931 M/sec                  
               263,446      branch-misses             #    0.09% of all branches        

           0.142652050 seconds time elapsed











