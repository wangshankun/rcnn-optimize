#include <iostream>
#include <algorithm> 
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sched.h>
#include <ctype.h>
#include <string.h>
#include <pthread.h>

#include "thread_pool.h"

using namespace std;

#define CYC_NUM 100000000
#define STEP 16

long func(volatile long* arg_num, int id, int test_step)
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //std::cout<<"worker thread ID:"<<std::this_thread::get_id()<<" arg_num: " << arg_num<<std::endl;
    //绑定CPU
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(id, &mask);
    if( sched_setaffinity(0, sizeof(mask), &mask) == -1)
    {
        fprintf(stderr, "WARNING: Could not set CPU Affinity \r\n"); 
    }
    else
    {
        //fprintf(stderr, "Bind process #%d to CPU #%d \r\n", id, id); 
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    //do some cpu expensive operation
    int cnt = CYC_NUM;
    while(cnt--)
    {
        arg_num[id * test_step]++;
    }
    //fprintf(stderr, "%d \r\n",arg_num[id]); 
    
    return arg_num[id * test_step];
}
 

class Counters
{
    private:
        volatile long* m_test_long_arrys; //volatile避免编译器优化
        ThreadPool* m_exec_pool;
        int   m_thread_num;
        int   m_test_step;

    public:
        Counters(int m_thread_num, int m_test_step):m_thread_num(m_thread_num),m_test_step(m_test_step)
        {
            m_test_long_arrys  = new long[m_thread_num * m_test_step];
            m_exec_pool        = new ThreadPool(m_thread_num);
        }

        void Test()
        {
            fill_n(m_test_long_arrys, m_thread_num * m_test_step, 0);

            long result = 0;
            double elapsed;
            struct timespec start, finish;
            clock_gettime(CLOCK_MONOTONIC, &start);

            vector<future<long>> res_vec;
            for(int i = 0; i < m_thread_num; i++)
            {
                auto res = m_exec_pool->enqueue(func, m_test_long_arrys, i, m_test_step);
                res_vec.push_back(move(res));
            }

            for(auto &res: res_vec)
            {
                result = result + res.get();
            }

            fprintf(stderr, "result: %ld \r\n",result); 

            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed = (finish.tv_sec - start.tv_sec);
            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            printf("avg elapsed time:%f\r\n",elapsed/m_thread_num);
    
            
        }
        
        ~Counters(){};
};

int main()
{
    Counters test8(8, 1);
    test8.Test();
}
