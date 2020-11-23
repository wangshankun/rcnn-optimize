#include "ops.h"
int main(int argc, char* argv[])
{
    int threadNum = atoi(argv[1]);
    int outside   = 64*75;
    int channel   = 7687;
    int topk      = 3;
    float* sf_int  = (float*)malloc(outside * channel * sizeof(float));
    float* sf_out  = (float*)malloc(outside * channel * sizeof(float));
    float* ag_out  = (float*)malloc(outside * 2 * topk * sizeof(float));

    double elapsed;
    struct timespec start, finish, sfinish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int b = 0; b < 100; b++){
    softmax(sf_int, sf_out, outside, channel, threadNum);}
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f \r\n",elapsed/100);
    for(int b = 0; b < 100; b++) {
    argmax(sf_out,  ag_out, outside, channel, topk, threadNum);}
    
    clock_gettime(CLOCK_MONOTONIC, &sfinish);
    elapsed = (sfinish.tv_sec - finish.tv_sec);
    elapsed += (sfinish.tv_nsec - finish.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f \r\n",elapsed/100);


    free(sf_int);
    free(sf_out);
    free(ag_out);

    return 0;
}
