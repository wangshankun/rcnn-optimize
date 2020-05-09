#include <stdio.h>
#include <time.h>

#define A_LEN  128 * 1024 * 1024
float arr[A_LEN];

void calc()
{
    int i, k, j;
    for (k = 1; k < 1028; k++)
    {
        clock_t begin = clock();
        for (j = 0; j < 10; j++)
        {
            for (i = 0; i < A_LEN; i += k)
            {
                arr[i] *= 3;
            }
        }
        clock_t elapsed = clock() - begin;

        fprintf(stderr, "using time: %f  k:%d \r\n",((double)elapsed / CLOCKS_PER_SEC), k);
    }
 
}

int main(void)
{
    calc();
    return 0;
}
