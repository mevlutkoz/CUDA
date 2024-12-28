//$nvcc -03 memTransfer.cu -o memTransfer
// nvprof ./memTransfer

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

	
	//pageable host memory
	float *h_a = (float *)malloc(nbytes);

	float *d_a;
	cudaMalloc((void **)&d_a, nbytes);

	for(unsgined int size = 0; i < isize; i++)
	{
		h_a[i] = 0.5f;
	} 

	cudaMemcpy(d_a, h_a, nbytes, cudaMempcyHostToDevice);
	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);


	cudaFree(d_a);
	free(h_a);

	cudaDeviceReset();

	return EXIT_SUCCESS;
}		