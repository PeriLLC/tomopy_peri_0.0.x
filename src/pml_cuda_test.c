#include "pml_cuda.h"
#include <stdio.h>

int main()
{
	if (test_cuda()==PERI_CALL_SUCCESS)
	{
		printf("test Cuda successed!\n");
		return 0;
	}
	printf("test Cuda failed..\n");
	return 1;
}