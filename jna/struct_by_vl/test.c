#include<stdio.h>
typedef struct Example4Struct {
	int val;
} Example4Struct;
Example4Struct example4_getStruct()
{
	Example4Struct sval;
	sval.val = 23;
	return sval;
}
