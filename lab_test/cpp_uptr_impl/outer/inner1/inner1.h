#ifndef INNER1_H_
#define INNER1_H_

#include "outer.h"

class Outer::Inner1
{
    public:
        void inDisplay();
        Inner1();
    private:
        static int m_In;
};
#endif
