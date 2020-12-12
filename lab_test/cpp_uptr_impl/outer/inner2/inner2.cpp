#include"inner2.h"
Outer::Inner2::Inner2()
{
    m_In = 2222;
}
void Outer::Inner2::inDisplay()
{
    std::cout<<m_In<<std::endl;  //
}
