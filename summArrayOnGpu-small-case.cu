//nvcc sumArrayOnGpu-small-case.cu -o addVector
//./addVector

#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)
{
	const cudaError_t error = call;
	if(error != cudaSuccess)
	{
		printf("Error: %s :%d \n", __FILE__, __LINE__);
		printf("code: %d, reason:%s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0;  i<N; i++)
	{
		if(abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match\n");
			printf("Host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if(match) printf("Arrays match \n\n");
}


void initialData(float *p, int size)
{
	time_t t;
	srand((unsigned) time(&time));

	for(int i = 0; i<size; i++)
	{
		p[i] = (float)(rand() & 0xFF)/10.0f;
	}
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for(int idx = 0; idx<N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void sumArraysOnGpu(float *A, float *B, float *C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}


int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	int dev = 0;
	cudaSetDevice(0);


	int nElem = 32;
	printf("vector size %d\n", nElem);

	size_t size = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	hostRef = (float *)malloc(size);
	gpuRef = (float *)malloc(size);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0 size);
	memset(gpuRef, 0, size);

	float *d_A, *d_b, *d_C;
	cudaMalloc((float **)&d_A, size);
	cudaMalloc((folat **)&d_B, size);
	cudaMalloc((float **)&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	dim3 block(nElem);
	dim3 grid(nElem/block.x);

	sumArraysOnGpu<<<grid,block>>>(d_A, d_B, d_C);
	printf("Execution configuration <<< %d %d>>>\n", grid.x, block.x);

	cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);

	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	checkResult(gpuRef, hostRef, nElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return(0);
}




