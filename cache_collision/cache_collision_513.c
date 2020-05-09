#include <stdio.h>
#include <time.h>

#define SAMPLES 1000
#define MATSIZE 513

int mat[MATSIZE][MATSIZE];

void transpose()
{
    int i, j, aux;

    for (i = 0; i < MATSIZE; i++) {
        for (j = 0; j < MATSIZE; j++) {
            aux = mat[i][j];
            mat[i][j] = mat[j][i];
            mat[j][i] = aux;
        }
    }
}

int main(void)
{
    int i, j;

    for (i = 0; i < MATSIZE; i++) {
        for (j = 0; j < MATSIZE; j++) {
            mat[i][j] = i + j;
        }
    }

    clock_t begin = clock();
    for (i = 0; i < SAMPLES; i++) {
        transpose();
    }
    clock_t elapsed = clock() - begin;

    printf("Average for a matrix of %d : %f s\n",
        MATSIZE, ((double)elapsed / CLOCKS_PER_SEC) / MATSIZE);

    return 0;
}
