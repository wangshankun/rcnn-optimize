#include "outer.h"
#include "inner1.h"
#include "inner2.h"

int Outer::Inner1::m_In=11;
int Outer::Inner2::m_In=22;

Outer::~Outer() = default;
Outer::Outer()
{
    inner1_uptr = std::make_unique<Inner1>();
    inner2_uptr = std::make_unique<Inner2>();
}

void Outer::exe_submodle1()
{
	inner1_uptr->inDisplay();
}

void Outer::exe_submodle2()
{
	inner2_uptr->inDisplay();
}
