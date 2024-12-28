/*
* $nvcc -arch=sm_35 -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt
* $ ./nestedHelloWorld		
*		
*
*		-rdc=true ===> forces the generation of relocatable device code a requirements for dynamic parallelism	
*
*
*/



__global__ void nestedHelloWorld(const int iSize, int iDepth)
{
	int tid = threadIdx.x;

	printf("Recursion: %d, Hello World from thread: %d block: %d\n" iDepth, tid, blockIdx);

	if(iSize == 1)
		return;

	int nthreads = iSize>>1;

	if(tid == 0 && nthreads > 0)
	{
		nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
		printf("========> nested execution depth : %d\n", iDepth);
	}
}




// $nvcc *arch=sm35 -rdc=true nestedReduce.cu -o nestedReduce -lcudadevrt
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize)
{
	unsignedi int tid = threadIdx.x;

	int *idata = g_idata + blokcIdx.x * blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	if(isize == 2 && tid == 0)
	{
		g_odata[blockIdx.x] = *(idata) + *(idata + 1);
		return;
	}

	int istride = isize >> 1;

	if(istride > 1 && tid < stride)
	{
		idata[tid] += idata[tid + stride];
	}
	__synchthreads();

	if(tid == 0)
	{
		gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
		cudaDeviceSynchronize();
	}

	__synchthreads();
}


__global__ void gpuRecursiveReduceNoSync(int *g_idata, int *g_odata, unsigned int isize)
{
	unsigned int tid = threadIdx.x;

	int *idata = g_idata + blockDimx.x * blockIdx.x;
	int *odata = g_odata + blockIdx.x;

	if(isize == 2 && tid == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}

	int stride = isize >> 1;

	if(stride > 1 && tid < stride)
	{
		idata[tid] += idata[0] + idata[1];
		if(tid == 0)
		{
			gpuRecursiveReduceNoSync<<<1, stride>>>(idata, odata, stride);
		}
	}
}

__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int istride, const int dim)
{
	int *idata = g_idata + blockIdx.x * dim;

	if(istride == 1 & threadIdx.x == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}

	idata[threadIdx.x] += idata[threadIdx.x + stride];

	if(threadIdx.x == 0 && blockIdx.x == 0)
	{
		gpuRecursiveReduce2<<<gridDim.x, stride/2>>>(g_idata, g_odata, stride/2, dim);
	}
}
