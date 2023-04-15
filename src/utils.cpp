#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// A utility function to swap to integers
static void swap(int *pA, int *pB)
{
    int temp = *pA;
    *pA = *pB;
    *pB = temp;
}

static void swap(float *pA, float *pB, int *pIdxA, int *pIdxB)
{
    float temp = *pA;
    *pA = *pB;
    *pB = temp;

    int tempIdx = *pIdxA;
    *pIdxA = *pIdxB;
    *pIdxB = tempIdx;
}

void randomize(int arr[], int n)
{
    srandom (time (0));
    
    for (int i = n - 1; i > 0; i--)
    {
        int j = random() % (i + 1);
 
        if (j != i)
            swap(&arr[i], &arr[j]);
    }
}

// Partition the array using the last element as the pivot
static int partition(float arr[],  int idxArr[], int low, int high)
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
void quickSortWithIdx(float arr[], int idxArr[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, idxArr, low, high);
        quickSortWithIdx(arr, idxArr, low, pi - 1);
        quickSortWithIdx(arr, idxArr, pi + 1, high);
    }
}