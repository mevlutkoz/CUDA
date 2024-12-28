//$nvcc -03 pinnedMemory.cu -o pinnedMemory
// nvprof ./pinnedMemory

#include <stdio.h>
#incldue <cuda_runtime.h>


__global__ void sumArrays(float *a, float *b, float *c, const int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}


int main(void)
{
	int dev = 0;
	cudaSetDevice(0);

	unsigned int isize = 1<<22;
	unsigned int nbytes = sizeof(float) * isize;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	printf("Starting: %s at", argv[0]);
	printf("device %d: memory size %d nbyte %5.2fMB\n", dev, deviceProp.name, isize, nbytes/(1024.0f * 1024.0f));

	

	float *h_a_pinned;
	cudaError_t status = cudaMallocHost((void **)&h_a_pinned, nbytes);
	if(status != cudaSuccess)
	{
		fprintf(stderr, "Error allocating pinned host memory.\n");
		fprintf(stderr, Error code: %d error string:%s\n, status, cudaGetErrorString(status));
		exit(1);
	} 

	float *d_a;
	cudaMalloc((void **)&d_a, nbytes);

	for(unsgined int size = 0; i < isize; i++)
	{
		h_a[i] = 0.5f;
	} 

	cudaMemcpy(d_a, h_a, nbytes, cudaMempcyHostToDevice);
	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);


	cudaFree(d_a);
	cudaFreeHost(h_a_pinned);

	cudaDeviceReset();

	return EXIT_SUCCESS;
}		