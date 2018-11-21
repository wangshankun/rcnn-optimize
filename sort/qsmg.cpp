//https://blog.csdn.net/NK_test/article/details/51037337

#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <errno.h>
#include <climits>
#include <stdlib.h>
#include <algorithm>
using namespace std;
 
//错误检查函数
inline void ERR_EXIT(const string &msg,int retnum)
{
    if(retnum!=0)
    {
        cerr<<msg<<": "<<strerror(retnum)<<endl;
        exit(EXIT_FAILURE);
    }
}
#define NUMMAX 3000000L
#define NTHR 4
#define TNUM (NUMMAX/NTHR)
long int nums[NUMMAX];
long int snums[NUMMAX];
pthread_barrier_t b;
 
void * workThread(void *args)
{
    long index=(long)args;
    sort(nums+index,nums+index+TNUM);
    pthread_barrier_wait(&b);
    pthread_exit(NULL);
}
 
void merge()
{
    long index[NTHR];
    for(long i=0;i<NTHR;i++)
        index[i]=i*TNUM;
 
    for(long i=0;i<NUMMAX;i++)
    {
         long min_index;
         long min_num=LONG_MAX;
         for(long j=0;j<NTHR;j++)
         {
             if((index[j]<(j+1)*TNUM)&&(nums[index[j]]<min_num))
             {
                 min_num=nums[index[j]];
                 min_index=j;
             }
         }
         snums[i]=nums[index[min_index]];
         index[min_index]++;
    }
 
}
int main()
{
    srandom(time(NULL));
    for(unsigned long i=0;i<NUMMAX;i++)
        nums[i]=random();
 
    struct timeval start,end;
    pthread_t tid;
    //计时开始
    gettimeofday(&start,NULL);
    pthread_barrier_init(&b,NULL,NTHR+1);
    for(unsigned long i=0;i<NTHR;i++)
        pthread_create(&tid,NULL,workThread,(void*)(i*TNUM));
 
    pthread_barrier_wait(&b);
    merge();
    gettimeofday(&end,NULL);
 
    //计算用时
    long long startusec=start.tv_sec*1000000+start.tv_usec;
    long long endusec=end.tv_sec*1000000+end.tv_usec;
    double elapsed=(double)(endusec-startusec)/1000000.0;
    printf("qsmg elapsed time:%f\r\n",elapsed);
    return 0;
}
