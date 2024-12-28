//nvcc -arch=sm_20 sumMatrixOnGpu-2D-grid-blokc.cu -o matrix
// ./matrix

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)
{
	const cudaError_t error =call;

	if(error != cudaSuccess)
	{
		printf("Error: %s, %d\n", __FILE__, __LINE__);
		printf("Error code: %d, error string %s\n",error, cudaGetErrorString(error));
		exit(-10 * error);
	}
}

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tpp.tv_usec * 1.e-6);
}


void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for(int iy=0; iy<ny; iy++)
	{
		for(int ix=0; ix<nx; i++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}

		ia += nx;
		ib += nx;
		ic += nx;
	}
}


void checkResult(float *A, float *B, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0; i<N; i++)
	{
		if(abs(A[i] - B[i]) > epsilon)
			{
				match = 0;
				printf("Arrays do no match\n");
				printf("Host: %5.2f, gpu: %5.2f at current:%d\n\n", *(A+i), *(B+i), i);
				break;
			}

	}
	if(match)
		printf("Arrays match\n\n");

}


__global__ void sumMatrixOnGpu2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;

	unsigned int idx = iy * nx + ix;

	if(iy < ny && ix < nx)
		MatC[idx] = MatA[idx] + MatB[idx];
}


int main( int argc, char ** argv)
{
	printf("%s...Starting\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device: %d - %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	int nx = 1<<14;
	int ny = 1<<14;
	int nxy = nx*ny;

	int nBytes = nxy * sizeof(float);

	printf("Matrix size: nx %d ny %d \n", nx,ny);

	float *hA, *hB, *hostRef, *gpuRef;

	hA = (float *)malloc(nBytes);
	hB = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuref = (float *)malloc(nBytes);

	double iStart= cpuSecond();
	initialData(hA, nxy);
	initialData(hB, nxy);
	double iElaps = cpuSecond() - iStart;

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	iStart = cpuSecond();
	sumMatrixOnHost(hA,hB,hostRef, nx,ny);
	iElaps = cpuSecond() - iStart;

	float *dA, *dB, *dC;
	cudaMalloc((float **)&dA, nBytes);
	cudaMalloc((float **)&dB, nBytes);
	cudaMalloc((float **)&dC, nBytes);

	cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice);

	int dimx = 32;
	int dimy = 32;

	dim3 block(dimx,dimy);
	dim3 grid((nx + block.x-1)/block.x), (ny + blokc.y -1)/block.y));

	iStart = cpuSecond();
	sumMatrixOnGpu2D<<<grid, block>>>(dA,dB,dC, nx,ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnGpu2D<<<(%d, %d), (%d, %d)>>> elapsed %f seconds \n", grid.x, grid.y, block.x, block.y, iElaps);

	cudaMemcpy(gpuRef, dC, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nxy);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(hA);
	free(hB);
	free(gpuRef);
	free(hostRef);


	cudaDeviceReset();

	return(0);


}