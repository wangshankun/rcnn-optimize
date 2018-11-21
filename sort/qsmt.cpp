//http://xiaoxia.org/2011/03/11/cc-multithreaded-programming-introduction-to-1-quicksort/
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <pthread.h>
#include <time.h>
#include <stdio.h>
using namespace std;

const int MaxData = 3000000;
const int ThreadCount = 4;
int data[MaxData];
pthread_t threads[ThreadCount];
int threadIndex = 0;
int region[ThreadCount][2];
pthread_mutex_t threadMutex;

int GetThreadNumber()
{
    /* 快速判断是否还有线程需要创建 */
    if(threadIndex >= ThreadCount)
        return -1;
    int threadNumber;
    pthread_mutex_lock(&threadMutex);
    /* 有可能在获得锁之后，threadIndex已经被其他线程修改过，所以务必检查一次 */
    if(threadIndex < ThreadCount)
        threadNumber = threadIndex ++;
    else
        threadNumber = -1;
    pthread_mutex_unlock(&threadMutex);
    return threadNumber;
}

void QuickSort(int begin, int end);
void* QuickSort_Procedure(void *data)
{
    //cout << ((int*)data)[0] << ", " << ((int*)data)[1] << endl;
    QuickSort(((int*)data)[0], ((int*)data)[1]);
    return 0;
}

void QuickSort(int begin, int end)
{
    int middle, i = begin, j = end;
    /* Pivot */
    if(threadIndex<ThreadCount)
        middle = (float)((begin + end) >> 1) / MaxData * 65535;
    else
        middle = data[(begin + end) >> 1];
    /* Iterating */
    do{
        while(data[i] < middle) i++;
        while(data[j] > middle) j--;
        if(i <= j) 
            swap(data[i++], data[j--]);
    }while(i < j);
    /* Divide and conquer */
    if(begin < j){
        int idx = GetThreadNumber();
        if(idx != -1){
            /* Create a thread to conquer the left part */
            region[idx][0] = begin, region[idx][1] = j;
            pthread_create(&threads[idx], 0, QuickSort_Procedure, region[idx]);
        }else{
            QuickSort(begin, j);
        }
    }
    if(i < end)
        QuickSort(i, end);
}

int main(int argc, char **argv)
{
    pthread_mutex_init(&threadMutex, NULL);
    for(int i=0; i<MaxData; i++)
        data[i] = rand()&0xffff;

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    QuickSort(0, MaxData-1);
    /* Wait for threads to end */
    for(int i=0; i<ThreadCount; i++)
        pthread_join(threads[i], 0);
    /*
    for(int i=0; i<MaxData; i++)
        cout << data[i] << " ";
    cout << endl;
    */
    pthread_mutex_destroy(&threadMutex);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("qsmt elapsed time:%f\r\n",elapsed);
    return 0;
}


