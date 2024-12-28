//reducing with a template functions
// call this kernel function with reduceCompleteUnrollTemplate<64><<<grid.x/8, block>>>(idata,odata,size)
//best reduction so far

template <unsigned int iBlckSize>
__global__ void reduceCompleteUnrollTemplate(int * g_idata, int *g_odata, unsigned int size)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blokcDim.x * blockIdx.x * 8 + threadIdx.x;

	int *idata = g_idata + blockDim.x * blokcIdx.x * 8;

	if(idx + 7 * blockDim.x < size)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blocDim.x];
		int a3 = g_idata[idx + blokcDim.x * 2];
		int a4 = g_idata[idx + blokcDim.x * 3];
		int b1 = g_idata[idx + blokcDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blokcDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];

		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}

	if(iBlckSize >= 1024 && tid < 512)
	{
		idata[tid] += idata[tid + 512];
	}	
	__synchthreads();

	if(iBlckSize >= 512 && tid < 256)
	{
		idata[tid] += idata[tid + 256];
	}
	__synchthreads();
	if(iBlckSize >= 256 && tid < 128)
	{
		idata[tid] += idata[tid + 128];
	}	
	__synchthreads();
	

	if(iBlckSize >= 128 && tid < 64)
	{
		idata[tid] += idata[tid + 64];
	}
	__synchthreads();

	if(tid < 32)
	{
		volatile int *vsmem = idata;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8]
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmsm[tid + 1];
	}


	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];

}



	
}





//reduce complete unroll while handling 8 data blokcs
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int size)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blokcDim.x * blockIdx.x * 8 + threadIdx.x;

	int *idata = g_idata + blockDim.x * blokcIdx.x * 8;

	if(idx + 7 * blockDim.x < size)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blocDim.x];
		int a3 = g_idata[idx + blokcDim.x * 2];
		int a4 = g_idata[idx + blokcDim.x * 3];
		int b1 = g_idata[idx + blokcDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blokcDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];

		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}

	__synchthreads();


	if(blockDim.x >= 1024 && tid < 512)
		idata[tid] += idata[tid + 512];
	__synchthreads();

	if(blokcDim.x >= 512 && tid < 256)
		idata[tid] += idata[tid + 256];
	__synchthreads();
	
	if(blockDim.x >= 256 && tid < 128)
		idata[tid] += idata[tid + 128];
	__synchthreads();

	if(blokcDim.x >= 128 && tid < 64)
		idata[tid] += idata[tid + 64];
	__synchthreads();

	if(tid < 32)
	{
		volatile int *vsmem = idata;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8]
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmsm[tid + 1];
	}


	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];

}




//Reduce with unrolled warps, with volatile keyword to allow the directly store the global
//memory rather than cache or register.
//Handle 8 data blocks

__global__ void reduceUnrollWarps8(int * g_idata, int *g_odata, unsigned int size)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blokcDim.x * 8 + threadIdx.x;

	int *idata = g_idata + blockDim.x * blokcIdx.x * 8;

	if(idx + 7.blokcDim.x < size)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blocDim.x];
		int a3 = g_idata[idx + blokcDim.x * 2];
		int a4 = g_idata[idx + blokcDim.x * 3];
		int b1 = g_idata[idx + blokcDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blokcDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];

		g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;		 
	}
	__synchthreads();

	for(int stride = blokcDim.x/2; stride > 32; stride >>=1)
	{
		if(tid < stride)
		{
			idata[tid] += idata[tid+stride];
		}

	__synchthreads();
	}

	if(tid < 32)
	{
		volatile int *vsmem = idata;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8]
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmsm[tid + 1];
	}


	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}


//reduce with unrolling
__global__ void reduceUnrolling2(int *indata, int *outdata, unsgined int n)
{
	unsigned int tid = threadIdx.x;
	unsgined int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	ind *data = blokcIdx.x * blockDim.x * 2;

	if(idx + blockDim.x < n)
		indata[idx] += indata[idx + blockDim.x]; 

	__synchthreads();
	
	for(int stride = blokcDim.x/2 ; stride > 0; stride >>=1)
	{	
		data[tid] += data[tid + stride];
	}

	__synchthreads();

	if(tid == 0)
		data[blockDim.x] = data[0];
}



//interleaved pair implementation with less divergence
__global__ void reduceInterleaved(int *idata, int *odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int *data = idata + blockDim.x * blockIdx.x;

	if(idx >= n)
		return;


	for(int stride = blockDim.x/2; stride > 0; stride>>1)
	{
		if(tid<strie)
		{
			data[tid] += data[tid + stride];
		}
	}

	__synchthreads();


	if(tid ==0)
		odata[blockIdx.x] = data[0];
}



//Neighbored implementation with less divergence
__global__ void reduceNeigboredLess(int *in, int *out, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int *idata = in + blockDim.x * blockIdx.x;

	if(idx >= n)
		return;

	for(stride = 1; stride < blockDim.x; stride*=2)
	{
			int index = stride * 2 * tid;
			if(index < blockDim.x)
			{
				idata[index] += idata[index + stride];
			}
		__synchthreads()
	}

	if(tid == 0)
		out[blockIdx.x] = idata[0];
}



//Unoptimized reduceNeighbored implementation
__global__ void reduceNeighborhed(int *in, int *out, unsigned int n)
{
	unsigned int tid = threadIdx.x
	unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

	int *idata = in + blockDim.x * blockIdx.x;

	if(idx > n)
	{
		return;
	}


	for(int stride = 1 ; stride < blockDim.x; stride *=2)
	{
		if(tid %(2*stride) == 0)
		{
			idata[i] += idata[tid + stride];
		}

		__synchtreads();
	}

	if(tid == 0)
		out[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, const int size)
{
	if(size == 0)
		return data[0];

	int const stride = size/2;

	for(int i = 0; i < size; i++)
		data[i] += data[i + stride]

	return recursiveReduce(int *data, stride);

}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s starting the reduction at", argc[0]);
	printf("Device: %d  %s", dev, deviceProp.name);
	cudaSetDevice(dev);

	bool bResult = false;
	int size = 1<<23;
	printf("With array size %d \n", size);

	int blocksize = 512;

	if(argc > 2)
		blocksize = atoi(argv[1]);

	dim3 block(blocksize,1);
	dim3 grid((size + block.x -1)/block.x, 1);
	printf("grid: %d block: %d \n", grid.x, block.x);

	size_t nbytes = sizeof(int) * size;

	int *h_indata = (int *)malloc(nBytes);
	int *h_odata = (int *)malloc(grid.x * sizeof(int));
	int *temp = (int *)malloc(nBytes);

	for(int i = 0; i < size; i++)
		h_indata[i] = (int)(rand() & 0xFF);

	memcpy(temp, h_indata, nbytes);


	size_t start, elaps;
	int gpu_sum = 0;

	int *d_indata = NULL;
	int *d_odata = NULL;

	cudaMalloc((void **)&d_indata, nbytes);
	cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

	start = cpuSecond();
	int cpu_sum = recursiveReduce(tmp,size);
	elaps = cpuSecond() - start;
	printf("cpu reduce elapsed: %d ms cpu_sum : %d\n", elaps, cpu_sum);

	cudaMemcpy(d_indata, h_indata, nbytes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	start = cpuSecond();
	warmup<<<grid,block>>>(d_indata, d_odata, size);
	elaps = cpuSecond() - start;

	cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;
	for(int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];

	printf("warmup kernel elapsed: %d gpu_sum: %d <<<grid: %d block: %d>>>\n", elaps, gpu_sum, grid.x, block.x);

	cudaMemcpy(d_indata, h_indata, nbytes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	start = cpuSecond();
	reduceNeighborhed<<<grid,block>>>(d_indata, d_odata, size);
	cudaDeviceSynchronize();
	elaps = cpuSecond() - start;

	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;
	for(int i = 0; i<grid.x; i++)
		gpu_sum+=h_odata[i];

	printf("gpuNeighbored elapsed: %d ms gpu_sum: %d <<< grid.x, block.x>>> \n", elaps, gpu_sum, grid.x, block.x);

	cudaDeviceSynchronize();

	free(h_odata);
	free(h_indata);
	free(temp);

	cudaFree(d_indata);
	cudaFree(d_odata);

	cudaDeviceReset();


	bResult = (gpu_sum == cpu_sum);
	if(!bResult)
	{
		printf("Test failed"):
	}

	return EXIT_SUCCESS;
}

