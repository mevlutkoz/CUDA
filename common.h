#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)
{
	const cudaError_t error = call;
	if(call != cudaSucces)
		{
			fprintf(stderr, "Error: %s: %d", __FILE__, __LINE__);
			fprintf(stderr, "Code: %d ==> reason: %s\n",error, cudaGetErrorString(error));
		}

inline double cpuSecond()
{
	struct timeval tp;
	struct timezone tzp;

	int i = gettimeofday(&rp, &tzp);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif