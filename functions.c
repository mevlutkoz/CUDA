#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

void checkResult(float *hostRef, float * gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0; i < N; i++)
	{
		if(abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match\n");
			printf("Host: %5.2f, Device: %5.2f at current: %d\n", hostRef[i], gpuRef[i], i);

			break;
		}
	}

	if(match) printf("Arrays matched\n");
}

void initialData(float *A, const int size)
{
	time_t t;
	srand((unsigned int) time(&t));

	for(int i = 0; i< N; i++)
		A[i] = (float) (rand() & 0xFF)/10.0f;
}


void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for(int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}






#endif