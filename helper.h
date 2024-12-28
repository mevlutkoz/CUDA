#ifndef _HELPER_H
#define _HELPER_H

void checkResult(float *hostRef, float * gpuRef, const int N);
void initialData(float *A, const int size);
void sumArraysOnHost(float *A, float *B, float *C, const int N);