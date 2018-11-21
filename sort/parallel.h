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
#define MAX_THREAD_NUMBER 4

#define MB
#define WMB
#define YIELDING    sched_yield()

typedef struct
{
    void (*routine)(void *);
    int  position;
    void* args;
} queue_t;

typedef struct
{
    queue_t*        queue;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} sub_thread_head_t;

static sub_thread_head_t   sub_thread_head[MAX_THREAD_NUMBER - 1];
static pthread_t           sub_thread_ids[MAX_THREAD_NUMBER - 1];

static void* sub_pthread_body(void *arg)
{

    int  pthread_pos = *(int *)arg;

    while(1)
    {
        pthread_mutex_lock(&sub_thread_head[pthread_pos].mutex);
        while (NULL == sub_thread_head[pthread_pos].queue)//使用while防止意外唤醒时候条件不满足
        {
            pthread_cond_wait(&sub_thread_head[pthread_pos].cond, &sub_thread_head[pthread_pos].mutex);//如果阻塞会自动释放mutexd
        }
        pthread_mutex_unlock(&sub_thread_head[pthread_pos].mutex);
        

        queue_t  *queue;
        queue = sub_thread_head[pthread_pos].queue;
        void (*routine)(void *) = queue -> routine;
        (routine)(queue -> args);
        
        pthread_mutex_lock(&sub_thread_head[pthread_pos].mutex);
        sub_thread_head[pthread_pos].queue = NULL;
        pthread_mutex_unlock(&sub_thread_head[pthread_pos].mutex);
    }
    return ((void *)0);
}

void sub_pthreads_setup(void)
{
    int pthread_pos;
    for(pthread_pos = 0; pthread_pos < MAX_THREAD_NUMBER - 1; pthread_pos++)
    {
        sub_thread_head[pthread_pos].queue  = NULL;
        pthread_mutex_init(&sub_thread_head[pthread_pos].mutex, NULL);
        pthread_cond_init (&sub_thread_head[pthread_pos].cond, NULL);

        pthread_create(&sub_thread_ids[pthread_pos], NULL, &sub_pthread_body, &pthread_pos);
    }
}

int using_sub_thread(queue_t* input_queue)
{

    int pthread_pos;
    for(pthread_pos = 0; pthread_pos < MAX_THREAD_NUMBER - 1; pthread_pos++)
    {
        pthread_mutex_lock(&sub_thread_head[pthread_pos].mutex);
        if (NULL == sub_thread_head[pthread_pos].queue)
        {
            input_queue -> position = pthread_pos;
            sub_thread_head[pthread_pos].queue = input_queue;
            pthread_cond_signal(&sub_thread_head[pthread_pos].cond);
            pthread_mutex_unlock(&sub_thread_head[pthread_pos].mutex);
            return 0;
        }
        pthread_mutex_unlock(&sub_thread_head[pthread_pos].mutex);
    }
    return -1;
}