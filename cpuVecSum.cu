
//nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sum
//Xcompiler gives command direclty to the C compiler, we send the command "Xcompiler -std=c99" to the C compiler

#incldue <stdio.h>
#include <stdlib.h>
#include <time.h>

void sumArraysOnHost(float *a, float *b, float *c, cont int N)
{
	for(int idx = 0; idx < N; idx++)
		c[idx] = a[idx] + b[idx];
}


void initData(float *a, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for(int i = 0; i < size; i++)
		a[i] = (float)(rand() & OxFF) / 10.0f;
}


int main(int argc, char **argv)
{
	int nBytes = 1024;
	float *h_A, *h_B, *h_C;

	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);

	initData(h_A, nBytes);
	initData(h_B, nBytes);

	sumArraysOnHost(h_A, h_B, h_C, nBytes);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}