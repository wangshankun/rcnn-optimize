#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include<thread>
#include <chrono>
using namespace std;

void notify(int signum)
{
    static int count = 0;
    cout << count++ << endl;
}

void init_sigaction()
{
    struct sigaction act;
    act.sa_handler = notify;
    act.sa_flags = 0;
    sigemptyset(&act.sa_mask);
    sigaction(SIGALRM, &act, NULL);
}

void init_time(struct itimerval* val)
{
    val->it_value.tv_sec = 1;
    val->it_value.tv_usec = 0;

    val->it_interval.tv_sec = 1;
    val->it_interval.tv_usec = 0;

    setitimer(ITIMER_REAL, val, NULL);
}

void uninit_time(struct itimerval* val)  
{  
    val->it_value.tv_sec = 0;  
    val->it_value.tv_usec = 0;  
    val->it_interval = val->it_value;  
    setitimer(ITIMER_REAL, val, NULL);  
}

int main()
{
    init_sigaction();

    struct itimerval val;
    init_time(&val);
    while(1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(6000));;
        uninit_time(&val);
        cout << "cancle timer" << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));;
        cout << "exit "<< endl;
        break;
    }

    return 0;
}
