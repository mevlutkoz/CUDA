//$nvcc -arch=sm_20 sumMatrixOnGpuMix.cu -o mat2D1D

#include <stdio.h>
#include cuda_runtime.h>

#define CHECK(call)
{
	const cudaError_t error = call;
	if(error != cudaSuccess)
	{
		printf("Error : %s %d\n", __FILE__, __LINE__);
		printf("Code: %d-> %s \", error, cudaGetErrorString(error));
		exit(error * -10);
	}
}



double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);	
}




void initialData(float *ip, const int size)
{
	int i;
	for(i = 0 ; i < size ; i++)
	{
		ip[i] = (float)(rand() & 0xFF)/10.0f;
	}

	return;
}


void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for(int iy = 0; iy<ny; iy++)
	{
		for(int ix = 0; ix<nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}

	return;
}





void checkResult(float *a, float *b, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0; i<N ; i++)
	{
		if(abs(a[i] - b[i] > epsilon)
		{
			match = 0;
			printf("host : %f, gpu: %f\n", a[i], b[i]);
			break;
		}
	}

	if(match)
		printf("Arrays are matched\n");
	else
	printf("Arrays do not match\n");
}


__global__ void sumMatrixOnGpuMix(float *A, float *B, float *C, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int idx = iy*nx + ix;

	if(ix < nx && iy < ny)
	{
		C[idx] = A[idx] + B[idx];
	}
}
int main(int argc, char **argv)
{
	printf("%s...Starting\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(%deviceProp, dev));
	printf("Using the device:%d -> %s\n", dev,deviceProp.name);
	CHECK(cudSetDevice(dev));

	int nx = 1<<14;
	int ny = 1<<14;

	int nxy = nx*ny;
	int nBytes = sizeof(float) * nxy;
	printf("Matrix size: nx: %d, ny:%d\n",nx,ny);

	float *hA, *hB, *hostRef, *gpuRef;
	hA = (float *)malloc(nBytes);
	hB = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes),


	double start = cpuSecond();
	initialData(hA, nxy);
	initialData(hB, nxy);
	double end = cpuSecond() - start;
	printf("initialize matrix elapsed: %f seconds\n", end);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	start = cpuSecond();
	sumMatrixOnHost(hA, hB, hostRef, nx,n,);
	end = cpuSecond() - start;
	printf("Sum matrix on host elapsed %f seconds\n", end);

	float *dA, *dB, *dC;
	cudaMalloc((float **)&dA, nBytes);
	cudaMalloc((float **)&dB, nBytes);
	cudaMalloc((float **)&dC, nBytes);


	CHECK(cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice));

	int dimx = 32;
	
	dim3 block(32);
	dim3 grid((nx + block.x -1)/ block.x, ny);

	start = cpuSecond();
	sumMatrixOnGPU1d<<<grid,block>>>(dA,dB,dC,nx,ny);
	CHECK(cudaDeviceSynchronize());
	end = cpuSecond() - start;
	printf("sumMatrixOnGPU1d<<<(%d, %d), (%d, %d)>>> elapsed: %f second\n", grid.x,grid.y,block.x,block.y, end);

	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(gpuRef, dC, nBytes, cudaMemcpyDeviceToHost));

	checkResult(hostRef, gpuRef, nxy);

	CHECK(cudaFree(dA));
	CHECK(cudaFree(dB));
	CHECK(cudaFree(dC));

	free(hA); free(hB); free(hC);

	CHECK(cudaDeviceReset());

	return (0);


}
