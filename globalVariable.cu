//nvcc -arch_sm=20 globalVariable.cu -o globalVariable
//./globalVariable

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


//Statically decleration of global variable using __device__ qualifier
//cudaMalloc decleares global variable dynamically
//cudaFree frees the dynamically allocated global device memory


__device__ float devData;

__global__ void checkGlobalVariable()
{
	printf("Device: the value of the gloval variable is %f\n", devData);

    //alter the value and force the added value to be float by using f suffix
	devData +=2.0f
}


int main(void)
{
	
	float value = 3.14f;

	
	//You have to initialize the global variable using the cudaMemcpyToSymbol
	cudaMemcpyToSymbol(devData, &vale, sizeof(float));
	printf("Host: copied %f to the global variable\n", value);

	checkGlobalVariable<<<1,1>>>();

	//You have to take the new value of global value from device using cudaMemcpyFromSymbol
	cudaMemcpyFromSymbol(&value, devData, sizeof(float));
	printf("Host: After kernel invocation new value of global value %f\n", value);

	cudaDeviceReset();

	return EXIT_SUCCESS;

}

/*

#include <cuda_runtime.h>
#include <iostream>

// Declare device global variable
__device__ float globalDeviceVariable;

__global__ void myKernel() {
    int tid = threadIdx.x;

    // Use atomicAdd to increment the global device variable atomically
    atomicAdd(&globalDeviceVariable, static_cast<float>(tid));
}

int main() {
    // Launching the kernel with 100 threads and 1 block
    myKernel<<<1, 100>>>();

    // Synchronizing the device to ensure completion of kernel execution
    cudaDeviceSynchronize();

    // Get the device global variable using cudaMemcpyFromSymbol
    float result = 0.0f;
    cudaMemcpyFromSymbol(&result, globalDeviceVariable, sizeof(float));

    // Print the result
    std::cout << "Value of the device global variable on the host: " << result << std::endl;

    return 0;
}

*/

