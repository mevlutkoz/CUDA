//$nvcc -03 zeroCopy.cu -o zeroCopy
//$nvprof ./zeroCopy


#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumArraysZeroCopy(float *a , float *b , float *c, const int size)
{
	unsigned int tid = threadId.x + blockDim.x * blockIdx.x;

	if(tid < size)
	{
		c[tid] = a[tid] + b[tid];
	}
}


__global__ void sumArrays(float *a , float *b , float *c, const int size)
{
	unsigned int tid = threadId.x + blockDim.x * blockIdx.x;

	if(tid < size)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main(void)
{
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	
	//Check wheter device support zero-copy mapped host memory

	if(!deviceProp.canMapHostMemory)
	{
		fprintf(stdout, "Device %d does not support mapping CPU host memory\n", dev);
		cudaDeviceReset();
		return EXIT_SUCCESS;
	}


	printf("Using device %d: %s\n", dev, deviceProp.name);

	int ipower = 10;
	if(argc > 1)
		ipower = atoi(argv[1]);

	int nElem = 1<<ipower;

	size_t nbytes = sizeof(float) * nElem;

	if(ipower < 18)
	{
		printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower, (float)nbytes/(1024.0f)).i
	}else{
		printf("Vectır size %d power %d nbytes %3.0f MB\n", nElem, ipower (float)nbytes/(1024.0f * 1024.0f));
	}

	float *ha, *hb, *host_ref, *gpu_ref;
	ha = (float *)malloc(nbytes);
	hb = (float *)malloc(nbytes);

	host_ref = (float *)malloc(nbytes);
	gpu_ref = (float *)malloc(nbytes);

	initialData(ha, nElem);
	initialData(hb, nElem);

	memset(host_ref, 0 ,nbytes);
	memset(gpu_ref, 0 ,nbytes);

	sumArrayOnHost(ha,hb,host_ref, nElem);

	float *da, *db, *dc;
	cudaMalloc((void **)&da, nbytes);
	cudaMalloc((void **)&db, nbytes);
	cudaMalloc((void **)&dc, nbytes);

	cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(ha, da, nbytes, cudaMemcpyHostToDevice);

	int iLen = 512;
	dim3 block (iLen);
	dim3 grid ((nElem + block.x -1)/block.x);

	sumArrays<<<grid, block>>>(da,db,dc, nElem);

	cudaMemcpy(gpu_ref, dc, nbytes, cudaMemcpyDeviceToHost);

	checkResults(host_ref, gpu_ref, nElem);
	
	cudaFree(da);
	cudaFree(db);
	free(ha);
	free(hb);

	unsigned int flags = cudaHostAllocMapped;
	cudaHostAlloc((void **)&ha, nbytes, flags);
	cudaHostAlloc((void **)&hb, nbytes, flags);

	initialData(ha, nElem);
	initialData(hb, nElem);
	memset(host_ref, 0, nbytes);
	memset(gpu_ref, 0, nbytes);

	cudaHostGetDevicePointer((void **)&da, (void *)ha, 0);
	cudaHostGetDevicePointer((void **)&db, (void *)hb, 0);

	sumArraysOnHost(ha, hb, host_ref, nElem);

	sumArraysZeroCopy<<<grid, block>>>(da,db,dc, nElem);
	cudaMemcpy(gpu_ref, dc, nbytes, cudaMemcpyDeviceToHost);

	checkResult(host_ref, gpu_ref, nElem);

	cudaFree(dc);
	cudaFreeHost(ha);
	cudaFreeHost(hb);

	free(host_ref);
	free(gpu_ref);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return EXIT_SUCCESS;
}