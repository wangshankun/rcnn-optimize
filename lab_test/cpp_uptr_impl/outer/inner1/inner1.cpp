#include"inner1.h"
Outer::Inner1::Inner1()
{
    m_In = 1111;
}
void Outer::Inner1::inDisplay()
{
    std::cout<<m_In<<std::endl;  //
}

