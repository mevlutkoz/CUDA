#include <stdio.h>
#include <stdlib.h>

#include "cudaHelperFunctions.h"



int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	CHECK(cudaSetDevice(dev));

	int nx = 1<<14;
	int ny = 1<<14;

	int nxy = nx*ny;
	size_t nBytes = nxy * sizeof(float);

	float *hA, *hB, *hostRef, *gpuRef;

	hA = (float *)malloc(nBytes);
	hB = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	size_t start = cpuSecond();
	initialData(hA, nxy);
	initialData(hB, nxy);
	size_t finish = cpuSecond() - start;

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	start = cpuSecond();
	sumMatrixOnHost(hA, hB, hostRef, nx , ny);
	finish = cpuSecond() - start;

	float *dA, float *dB, float *dC;
	CHECK(cudaMalloc((void **)&dA, nBytes));
	CHECK(cudaMalloc((void **)&dB, nBytes));
	CHECK(cudaMalloc((float **)&dC, nBytes));

	CHECK(cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice));

	int dimx = 32;
	int dimy = 32;

	if(argc > 2)
	{
		dimx = atoi(argv[1]);
		dimy = atoi(argv[2]);
	}

	dim3 block(dimx, dimy);
	dim3 grid((nx * block.x -1)/ block.x, (ny * block.y-1)/block.y);

	CHECK(cudaDeviceSynchronize());

	start = cpuSecond();
	sumMatrixOnGPU2D<<<grid,block>>>(dA,dB,dC,nx,ny);
	CHECK(cudaDeviceSynchronize());
	finish = cpuSecond() - start;

	printf("sumMatrixOnGPU2D<<<(%d, %d),(%d, %d)>>> elapsed: %d ms\n", grid.x, grid.y, block.x, block.y, finish);

	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, dC, nBtes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nxy);

	CHECK(dA);
	CHECK(dB);
	CHECK(dC);

	free(hA);
	free(hB);
	free(hostRef);
	free(gpuRef);

	CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;

}