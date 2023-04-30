#include <stdio.h>
#include <assert.h>
//#include <math.h>

#include "cuda_runtime.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "Utilities.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG

#define PI_R         3.14159265358979323846f

/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ static void swap(int *pA, int *pB)
{
    int temp = *pA;
    *pA = *pB;
    *pB = temp;
}

__device__ static void swap(float *pA, float *pB, int *pIdxA, int *pIdxB)
{
    float temp = *pA;
    *pA = *pB;
    *pB = temp;

    int tempIdx = *pIdxA;
    *pIdxA = *pIdxB;
    *pIdxB = tempIdx;
}

__device__ void randomize(int arr[], int n)
{
    curandState state;

    
    for (int i = n - 1; i > 0; i--)
    {
        curand_init(123123213213LL, i, 0, &state);
        int j =  (int)ceilf(curand_uniform(&state)) % (i + 1);
 
        if (j != i)
            swap(&arr[i], &arr[j]);
    }
}

// Partition the array using the last element as the pivot
__device__ static int partition(float arr[],  int idxArr[], int low, int high)
{
    float pivot = arr[high];
    int i = (low - 1);
  
    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(&arr[i], &arr[j], &idxArr[i], &idxArr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high], &idxArr[i + 1], &idxArr[high]);
    return (i + 1);
}
 
// Function to implement Quick Sort
__device__ void quickSortWithIdx(float arr[], int idxArr[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, idxArr, low, high);
        quickSortWithIdx(arr, idxArr, low, pi - 1);
        quickSortWithIdx(arr, idxArr, pi + 1, high);
    }
}