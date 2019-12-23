#include<stdio.h>
typedef struct Example3Struct {
	int val;
} Example3Struct;
void example3_sendStruct(const Example3Struct* sval)
{
	// note: printfs called from C won't be flushed
	// to stdout until the Java process completes
	printf("(C) %d\n", sval->val);
}
