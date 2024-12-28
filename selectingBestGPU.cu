#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
	int numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	
	if(numDevices > 1)
	{
		int maxMultiProcessors = 0, maxDevice = 0;
		for(int device = 0; device < numDevices; device++)
		{
			cudaDeviceProp props,
			cudaGetDeviceProperties(&props, device);

			if(maxMultiProcessors < props.multiProcessorCount)
			{
				maxMultiProcessors = props.multiProcessorCount;
				maxDevice = device;
			}

		}
		cudaSetDevice(maxDevice);
	}


	return 0;
}
