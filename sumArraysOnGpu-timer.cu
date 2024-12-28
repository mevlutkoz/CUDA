#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define CHECK(call)
{
	const cudaError_t error = call;
	if(error != cudaSuccess)
	{
		printf("Error %s: %d \n", __FILE__, __LINE__);
		printf("code %d, reason %s \n", error, cudaGetErrorString(error));
		exit(1);
	} 
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0 i < N; i++)
	{
		if(abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match\n");
			printf("Host: %5.2f gpu: %5.2f at current: %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if(match)
	{
		printf("Arrays match \n\n");
	}
}


void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned) time(&t));
	for(int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF)/10.0f;
	}	
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for(int idx = 0; idx<N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}


__global__ void sumArraysOnGpu(float *A, float *B, float *C, const int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < N)
	{
		C[idx] = A[idx] + B[idx];
	}
}


double cpuSecond(void)
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main(int argc, char **argv)
{
	printf(",%s Starting...\n",argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device: %d %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(device));

	int nElem = 1<<24;
	printf("Vector size %d\n", nElem);

	size_t bytes = sizeof(float) * nElem;

	float *hA, *hB, *hostRef, *gpuRef;

	hA = (float *)malloc(bytes);
	hB = (float *)malloc(bytes);
	hostRef	= (float *)malloc(bytes);
	gpuRef = (float * )malloc(bytes);

	double iStart, iElaps;
	iStart = cpuSecond();
	initialData(hA, nElem);
	initialData(hB, nElem);
	iElaps = cpuSecond() - iStart;

	memset(hostRef, 0, bytes);
	memset(gpuRef, 0, bytes);

	iStart = cpuSecond();
	sumArraysOnHost(hA,hB,hostRef, nElem);
	iElaps = cpuSecond() - iStart;

	float *dA, *dB, *dC;
	cudaMalloc((float **)&dA, bytes);
	cudaMalloc((float **)&dB, bytes);
	cudaMalloc((float **)&dC, bytes);

	cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

	int iLen = 1024;
	dim3 block(iLen);
	dim3 grid((nElem + block.x -1)/ block.x);

	iStart = cpuSecond();
	sumArrayOnGpu<<<grid, block>>>(dA,dB,dC,nElem);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("sumArrayOnGpu<<<%d, %d>>> Time elapsed: %f sec \n", grid.x,block.x, iElaps);

	cudaMemcpy(gpuRef, dC, bytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElems);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(hA);
	free(hB);
	free(hC);

	return(0);
}
