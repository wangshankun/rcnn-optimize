#define  _GNU_SOURCE
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <error.h>
#include <fcntl.h>
#include <float.h>
#include <poll.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <alloca.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/cdefs.h>

//根据CPU核数配置线程数
#define MAX_CPU_NUMBER 2

#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
#define MB
#define WMB
#define YIELDING    sched_yield()

typedef struct {
    void (*routine)(void *, int);
    int  position;
    void* args;
} queue_t;

typedef struct {
    queue_t * volatile queue  __attribute__((aligned(16)));
    volatile long status;
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
} sub_thread_status_t;

void all_sub_pthread_exec(queue_t *queue_Q, int exec_thread_num);
void sub_pthreads_setup(void);
