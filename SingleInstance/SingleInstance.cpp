#include <iostream> // std::cout
#include <mutex>    // std::mutex
#include <pthread.h> // pthread_create

class Single
{

public:
    // 获取单实例对象
    static Single &GetInstance();

    // 打印实例地址
    void Print();

private:
    // 禁止外部构造
    Single();

    // 禁止外部析构
    ~Single();

    // 禁止外部复制构造
    Single(const Single &signal);

    // 禁止外部赋值操作
    const Single &operator=(const Single &signal);
};

Single &Single::GetInstance()
{
    // 局部静态特性的方式实现单实例
    static Single signal;
    return signal;
}

void Single::Print()
{
    std::cout << "我的实例内存地址是:" << this << std::endl;
}

Single::Single()
{
    std::cout << "Single 构造函数" << std::endl;
}

Single::~Single()
{
    std::cout << "Single 析构函数" << std::endl;
}

void *PrintHello(void *threadid)
{
    // 主线程与子线程分离，两者相互不干涉，子线程结束同时子线程的资源自动回收
    pthread_detach(pthread_self());

    // 对传入的参数进行强制类型转换，由无类型指针变为整形数指针，然后再读取
    int tid = *((int *)threadid);

    std::cout << "Hi, 我是线程 ID:[" << tid << "]" << std::endl;

    // 打印实例地址
    Single::GetInstance().Print();

    pthread_exit(NULL);
}

#define NUM_THREADS 5 // 线程个数

int main(void)
{
    pthread_t threads[NUM_THREADS] = {0};
    int indexes[NUM_THREADS] = {0}; // 用数组来保存i的值

    int ret = 0;
    int i = 0;

    std::cout << "main() : 开始 ... " << std::endl;

    for (i = 0; i < NUM_THREADS; i++)
    {
        std::cout << "main() : 创建线程:[" << i << "]" << std::endl;

        indexes[i] = i; //先保存i的值

        // 传入的时候必须强制转换为void* 类型，即无类型指针
        ret = pthread_create(&threads[i], NULL, PrintHello, (void *)&(indexes[i]));
        if (ret)
        {
            std::cout << "Error:无法创建线程," << ret << std::endl;
            exit(-1);
        }
    }

    pthread_join(threads[NUM_THREADS-1], NULL);
    std::cout << "main() : 结束! " << std::endl;
    return 0;
    // 进程退出释放单实例的资源
}
