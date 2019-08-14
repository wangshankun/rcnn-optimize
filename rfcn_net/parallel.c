#include "parallel.h"
static sub_thread_status_t sub_thread_status[MAX_CPU_NUMBER - 1] __attribute__((aligned(32)));
static pthread_t           sub_threads_ids  [MAX_CPU_NUMBER - 1];
static int                 POS[MAX_CPU_NUMBER - 1];

static void pthread_bind(int cpu)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if(sched_setaffinity(0, sizeof(mask), &mask) == -1)
    {
        printf("set affinity failed..");
    }
}

static void* sub_pthread_body(void *arg)
{
    queue_t  *queue;
    int  pthread_pos = *(int *)arg;
//   如果cpu核数不超过8个,可以不用绑定cpu来提升单任务
//    pthread_bind(pthread_pos + 1);

    while(1)
    {
        pthread_mutex_lock(&sub_thread_status[pthread_pos].lock);
        if (!sub_thread_status[pthread_pos].queue)
        {
            sub_thread_status[pthread_pos].status = THREAD_STATUS_SLEEP;

            while (sub_thread_status[pthread_pos].status == THREAD_STATUS_SLEEP)
            {
                pthread_cond_wait(&sub_thread_status[pthread_pos].wakeup, &sub_thread_status[pthread_pos].lock);
            }
        }
        pthread_mutex_unlock(&sub_thread_status[pthread_pos].lock);
        queue = sub_thread_status[pthread_pos].queue;
        void (*routine)(void *, int) = queue -> routine;
        (routine)(queue -> args, queue -> position);

        sub_thread_status[pthread_pos].queue = (queue_t * volatile) ((long)sub_thread_status[pthread_pos].queue & 0);
    }
    return ((void *)0);
}

void all_sub_pthread_exec(queue_t *queue_Q, int exec_thread_num)
{
    int pthread_pos, i;
    //pthread_bind(0);
    if(exec_thread_num < 2) return;
    if(exec_thread_num > MAX_CPU_NUMBER) exec_thread_num = MAX_CPU_NUMBER;
    for(pthread_pos = 0; pthread_pos < exec_thread_num - 1; pthread_pos++)
    {
        if(queue_Q[pthread_pos + 1].routine)//从queue_Q[1]开始，这是唤醒子程序过程，而子程序从1开始计数
        {
            WMB;
            sub_thread_status[pthread_pos].queue = &(queue_Q[pthread_pos + 1]);
            WMB;
            if (sub_thread_status[pthread_pos].status == THREAD_STATUS_SLEEP)
            {
                pthread_mutex_lock  (&sub_thread_status[pthread_pos].lock);
                sub_thread_status[pthread_pos].status = THREAD_STATUS_WAKEUP;
                pthread_cond_signal(&sub_thread_status[pthread_pos].wakeup);//唤醒的子程序在死循环的sub_pthread_body中(routine)(queue -> args, queue -> position)执行
                pthread_mutex_unlock(&sub_thread_status[pthread_pos].lock);
            }
        }
    }
    if(queue_Q[0].routine)//queue_Q[0]是从主程序继承而来，接着主程序逻辑运行，并在最后while等待其他子程序归来
    {
        void (*routine)(void *, int) = queue_Q[0].routine;
        (routine)(queue_Q[0].args, queue_Q[0].position);
        for (i = 0; i < MAX_CPU_NUMBER - 1; i++)  while (sub_thread_status[i].queue) {YIELDING;};
    }
}

void sub_pthreads_setup(void)
{
    int pthread_pos;
    for(pthread_pos = 0; pthread_pos < MAX_CPU_NUMBER - 1; pthread_pos++)
    {
        POS[pthread_pos] = pthread_pos;
        sub_thread_status[pthread_pos].queue  = (queue_t *)NULL;
        sub_thread_status[pthread_pos].status = THREAD_STATUS_WAKEUP;

        pthread_mutex_init(&sub_thread_status[pthread_pos].lock, NULL);
        pthread_cond_init (&sub_thread_status[pthread_pos].wakeup, NULL);

        pthread_create(&sub_threads_ids[pthread_pos], NULL, &sub_pthread_body, &(POS[pthread_pos]));
    }
}
