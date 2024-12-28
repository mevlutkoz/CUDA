#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)
{
	const cudaError_t error = call;

	if(error != cudaSuccess)
	{
		printf("Error: %s, %d\n", __FILE__, __LINE__);
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(-10*error);

	}
}

void initialData(float *array, int N)
{
	for(int idx = 0; idx < N; idx++)
		*(array+i) = i;
}

void printMatrix(int *C, const int nx, const int ny)
{
	int *ic = C;

	printf("\nMatrix :(%d.%d)\n", nx,ny);
	for(int iy = 0; iy < ny; iy++)
	{
		for(int ix = 0; ix < nx; ix++)
		{
			printf("%3d\n", ic[ix]);
		}

		ic += nx;
		printf("\n");
	}

	printf("\n");

}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockDim.x + blockIdx.x;
	int iy = threadIdx.y + blockDim.y + blockIdx.y;

	unsigned int idx = iy * nx + ix;
	printf("thread_id:(%d, %d) blockId:(%d %d) coordinate (%d %d)"
		"global index:(%3d) ival: %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix,iy,idx, A[idx]);
}


int main(int argc, char **argv)
{
	printf("%s...Starting\n",argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device: %d:%s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	int nx = 8;
	int ny = 6;
	int nxy = 8*6;
	int nBytes = sizeof(float) * nxy;

	int *h_A = (float *)malloc(nBytes);
	initialData(h_A, nxy);
	printMatrix(h_A, nx,ny);

	int *d_MatA;
	cudaMalloc((float **)&d_MatA, nBytes);

	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

	dim3 block(4,2);
	dim3 grid((nx + block.x-1)/block.x, (ny + block.y -1)/block.y);

	printThreadIndex<<<grid,block>>>(d_MatA, nx,ny);
	cudaDeviceSynchronize();

	cudaFree(d_MatA);
	free("h_A");

	cudaDeviceReset();

	return(0);
}

