#ifndef OUTER_H_
#define OUTER_H_

#include <iostream>
#include <memory>
using namespace std;

class Outer
{
    public:
        Outer();
	~Outer();
        void exe_submodle1();
        void exe_submodle2();
    private:
        class Inner1;//预声明，其它人去实现
        std::unique_ptr<Inner1> inner1_uptr;//给inner类一个成员指针，让能够被调用
        //这样好处是代码解耦合

        class Inner2;//预声明，其它人去实现
        std::unique_ptr<Inner2> inner2_uptr;//给inner类一个成员指针，让能够被调用
        //这样好处是代码解耦合
};

#endif
