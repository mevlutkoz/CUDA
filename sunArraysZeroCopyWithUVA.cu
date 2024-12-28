#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Dont forget to include header file and .c file for extra functions
//$nvcc -03 -arch_sm=20 sumArrayZeroCopyWithUVA.cu -o sumArrayZeroCopyWithUVA
//$nvprof ./sumArrayZeroCopyWithUVA


__global__ void sumArraysZeroCopyWithUVA(float *a, float *b, float *c, const int N)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < N)
	{
		c[i] = a[i] + b[i];
	}
}


int main(int argc, char **argv)
{
	int dev = 0;

	CHECK(cudaSetDevice(dev));

	cudaDeviceProp deviceProp
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));

	if(!deviceProp.canMapHostMemory)
	{
		printf("Device does not support mapping CPU host memory");
		CHECK(cudaResetDevice());
		exit(EXIT_SUCCESS);
	}

	printf("Using the device %d : %s \n", dev, deviceProp.name);

	int ipower = 18;
	if(argc > 2)
		ipower = atoi(argv[1]);
	int nElem = 1<<ipower;
	size_t nBytes = sizeof(float) * nElem;

	if(ipower < 18)
	{
		printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower, (float)nBytes/(1024.f));
	}
	else{
		printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, ipower, (float)nBytes/(1024.0f * 1024.0f));		
	}

	float *ha, *hb, *hostref, *gpuref;

	ha = (float *)malloc(nBytes);
	hb = (float *)malloc(nBytes);
	hostref = (float *)malloc(nBytes);
	gpuref = (float *)malloc(nBytes);

	initialData(ha, nElem);
	initialData(hb, nElem);
	memset(hostref, 0, nBytes);
	memset(gpuref, 0, nBytes);

	sumArraysOnHost(ha, hb, hostref, nElem);

	float *da, *db, *dc;
	CHECK(cudaMalloc((float **)&da, nBytes));
	CHECK(cudaMalloc((float **)&db, nBytes));
	CHECK(cudaMalloc((float **)&dc, nBytes));

	CHECK(cudaMemcpy(da, ha, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(db, hb, nBytes, cudaMemcpyHostToDevice));

	int iLen = 512;
	dim3 block(iLen);
	dim3 grid((iLen + block.x -1) / block.x);

	double start = seconds();
	sumArrays<<<grid, block>>>(da, db, dc, nElem);
	CHECK(cudaDeviceSynchronize());
	double elapsed = seconds() - start;
	printf("sumArrays elapsed: %f s\n", elapsed);

	CHECK(cudaMemcpy(gpuref, dc, nBytes, cudaMemcpyDeviceToHost));
	checkResults(hostref, gpuref, nElem);

	CHECK(cudaFree(da));
	CHECK(cudaFree(db));

	free(ha);
	free(hb);

	//Zero copy without unified virual addressing which means compute capability is => lower than 4.0 (< 4.0)

	CHECK(cudaHostAlloc((void **)&ha, nBytes, cudaHostAllocMapped));
	CHECK(cudaHostAlloc((void **)&hb, nBytes, cudaHostAllocMapped));

	initialData(ha, nElem);
	initialData(hb, nElem);
	memset(hostref, 0, nBytes);
	memset(gpuref, 0, nBytes);

	CHECK(cudaHostGetDevicePointer((void **)&da, (void *)ha, 0));
	CHECK(cudaHostGetDevicePointer((void **)&db, (void *)hb, 0));

	start = seconds();
	sumArraysOnHostZeroCopy<<<grid, block>>>(da, db, dc, nElem);
	CHECK(cudaDeviceSynchronize());
	elapsed = seconds() - start;
	printf("Elapsed time in kernel with zero-copy: %lf\n", elapsed);

	CHECK(cudaMemcpy(gpuref, dc, nBytes, cudaMemcpyDeviceToHost));
	checkResults(gpuref, dc, nElem);

	//if we have a device which has compute capability is larger than 4.0, we can use UNIFIED VIRTUAL ADRESSING
	//when we allocate memory with cudaHostAlloc with flag = cudaHostAllocMapped it automatically becomes UVA

	memset(gpuref, 0, nBytes);
	start = seconds();
	sumArraysZeroCopyWithUVA<<<grid, block>>>(ha, hb, dc, nElem);
	CHECK(cudaDeviceSynchronize());
	elapsed = seconds() - start;
	printf("Elapsed time in kernel with Unified Virtual Adressing: %lf\n", elapsed);

	CHECK(cudaMemcpy(gpuref, dc, nBytes, cudaMemcpyDeviceToHost));
	checkResults(gpuref, hostref, nElem);

	CHECK(cudaFreeHost(ha));
	CHECK(cudaFreeHost(hb));
	CHECK(cudaFree(dc));

	free(hostref);
	free(gpuref);

	CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}

