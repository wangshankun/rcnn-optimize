//https://blog.csdn.net/Hairy_Monsters/article/details/79776744
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
 
void partition(int arr[], int low, int high, int *pos)
{
    int key = arr[low];
    int i = low, j = high;
    while(i < j)
    {
        while(i < j && arr[j] > key) j--;
        if(i < j)
        {
            arr[i++] = arr[j];
        }
        while(i < j && arr[i] < key) i++;
        if(i < j)
        {
            arr[j--] = arr[i];
        }
    }
    arr[i] = key;
    *pos = i;
}
 
int topK(int arr[], int low, int high, int k)
{
    int pos =0;
    partition(arr, low, high, &pos);
    int num = high - pos + 1;
    int index = -1;
    if(num == k)
    {
        index = pos;
    }
    else if(num > k)
    {
        index = topK(arr, pos + 1, high, k);
    }
    else
    {
        index =  topK(arr, low, pos -1, k - num);
    }
    return index;
}
 
 
#define NUMMAX   3000000
#define TOPK     1000
int data[NUMMAX];

int main()
{
    srandom(time(NULL));
    for(unsigned long i = 0;i < NUMMAX; i++)
    {
        data[i]=random();
    }
    
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int pos;
    pos = topK(data, 0, NUMMAX, TOPK);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("topk elapsed time:%f\r\n",elapsed);
   //for(unsigned long i = 0;i < NUMMAX;i++)
   //{
   //    printf("%d\n", data[i]);
   //}
   //printf("\n");
}
