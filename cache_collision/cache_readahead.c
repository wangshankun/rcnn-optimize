// big array; A[SIZE]
// we traverse in blocks; block i = i*skip mod SIZE
// if i=1, this is linear traversal
// if i=2, we should double the number of cache misses.
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long microsec() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return 1000000*t.tv_sec + t.tv_usec;
}


int BIGPRIME = 1000003;
int HEIGHT= 8000;
int LINESIZE = 64;
int INTSperLINE;		// should be 16
int BSIZE;			// # of bytes
int ASIZE;			// # of ints 

// hits blocks in order 0*skip, 1*skip, 2*skip, ...
void skipinit(int skip, int * A) {
	int blocknum = 0; 
	do {
		int bstart = blocknum*INTSperLINE;
		for (int j=bstart; j<bstart + INTSperLINE; j++) A[j] = 0;
		blocknum +=skip;
		if (blocknum>=BIGPRIME) blocknum -= BIGPRIME;
	} while (blocknum != 0);

} 
void main(int argc, char **argv) {
    INTSperLINE = LINESIZE/sizeof(int);		// should be 16
    BSIZE = BIGPRIME * LINESIZE;		// # of bytes
    ASIZE = BSIZE / sizeof(int);		// # of ints 
    int skip = 1;
    if (argc == 2) {
        skip=atoi(argv[1]);
    }
    printf("skip=%d\n", skip);

    int * A = (int*) malloc(BSIZE);

    /* */
    for (int i=0; i<ASIZE; i++) {
        A[i] = 0;
    }
    /* */
    long start, stop;
    long rowsum, colsum = 0;

    start = microsec();
    skipinit(skip, A);
    stop = microsec();
    printf("skip time: %ld\n", stop-start);
    //rowsum += stop-start;

    int RUNS = 50;

    for (int i=0; i<RUNS; i++) {
        start = microsec();
        skipinit(skip, A);
        stop = microsec();
        printf("skip time: %ld\n", stop-start);
        rowsum += stop-start;
    }

    printf("avg: %ld\n", rowsum/RUNS);

}
    
