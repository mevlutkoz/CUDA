#include <stdio.h>

#define N 1048576

__global__ void VecAdd(double *a, double *b, double *c,)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < N)
	{
		c[i] = a[i] + b[i];	
	}

}

int main(void)
{
	size_t bytes = sizeof(double) * N;

	double *a = (double *)malloc(bytes);
	double *b = (double *)malloc(bytes);
	double *c = (double *)malloc(bytes);

	double *d_A, *d_B, *d_C;

	cudaMalloc((double **)&d_A, bytes);
	cudaMalloc((double **)&d_B, bytes);
	cudaMalloc((double **)&d_C, bytes);

	for(int i = 0; i< N; i++)
	{
		a[i] = 1.0;
		b[i] = 2.0;
	}

	cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, b, size, cudaMemcpyHostToDevice);


	int threadsPerBlock = 256;
	int blocksPerGrid = ceil(float(N)/threadsPerBlock);

	VecAdd<<<blocksPerGrid, threadsPerBlokc>>>(d_A, d_B, d_C);
	cudaMemcpy(c, d_C, size, cudaMemcpyDeviceToHost); 

	double tolerence = 1.0e-14;
	for(int i = 0; i<N; i++)
	{
		if(fabs(c[i]) - 3.0) > tolerence)
		{
			printf("\nError: value of c[%d] = %d instead of 3.0 \n\n");
			exit(1);
		}
	}


	free(a);
	free(b);
	free(c);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);


	printf("\n------------------------\n");
	printf("__SUCCESS__\n");
	printf("--------------------------\n");
	printf("N                        =\n", N);
	printf("Threads per block = %d\n", threadsPerBlokc);
	printf("Blocks in grid = %d\n", blocksPerGrid);
	printf("---------------------------\n");


	return 0;
}