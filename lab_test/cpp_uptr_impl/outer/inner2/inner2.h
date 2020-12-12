#ifndef INNER2_H_
#define INNER2_H_

#include "outer.h"

class Outer::Inner2
{
    public:
        void inDisplay();
        Inner2();
    private:
        static int m_In;
};
#endif
