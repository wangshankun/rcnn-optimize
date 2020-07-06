#include <iostream>
#include <numeric>
#include <vector>

using namespace std;
// g++ -std=c++11 test.cpp && ./a.out
// C++ 运行时类型识别
// https://blog.csdn.net/smilestone322/article/details/23677325
// https://stackoverflow.com/questions/307352/g-undefined-reference-to-typeinfo
// https://blog.csdn.net/ai2000ai/article/details/47152133

class Employee 
{
public:
    virtual int salary() {};//虚函数需要实现，需要typeinfo，或者编译时候加上-frtti选项
};

class Manager : public Employee
{
public: 
    int salary();
};

class Programmer : public Employee
{
public:
    int salary();
    int bonus();//直接在这里扩展
};

int Manager::salary()
{
    cout << "Manager salary " << endl;
    return 0;
}

int Programmer::salary()
{
    cout << "Programmer salary " << endl;
    return 0;
}

int Programmer::bonus()
{
    cout << "Programmer Bonus " << endl;
    return 0;
}

class MyCompany
{
public:
    void payroll(Employee *pe);
};

void MyCompany::payroll(Employee *pe)
{
    Programmer *pm = dynamic_cast<Programmer *>(pe);
    
    //如果pe实际指向一个Programmer对象,dynamic_cast成功，并且开始指向Programmer对象起始处
    if(pm)
    {
        pm->bonus();
        //call Programmer::bonus()
    }
    //如果pe不是实际指向Programmer对象，dynamic_cast失败，并且pm = 0
    else
    {
        //use Employee member functions
        cout <<"Is Not Programmer" << endl;
    }
}

int main()
{
    MyCompany  company_test;
    Programmer pm;
    company_test.payroll(&pm);
    
    Manager mg;
    company_test.payroll(&mg);
    return 0;
}
