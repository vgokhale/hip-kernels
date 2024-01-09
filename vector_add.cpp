#include <assert.h>
#include <random>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

// Macro for checking errors in CUDA API calls

// Size of array.
#define N 110*256
#define ITER 102400

// Kernel
__global__ void shared_atomic_add(float *a, float *b, float offset)
{
    __shared__ float block[256];
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    block[tid] = a[id];
    for(int i = 0; i < ITER; ++i) {
        atomicAdd(&block[tid], offset);
    } 
    b[id] = block[tid];
}

// Main program
int main()
{
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(float);

    // Allocate memory for arrays A, B, and C on host
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    float *d_A, *d_B;
    HIP_ASSERT( hipMalloc(&d_A, bytes) );
    HIP_ASSERT( hipMalloc(&d_B, bytes) );

    // Fill host arrays A and B
    std::random_device rand;
    std::mt19937 mt(rand());
    std::uniform_real_distribution<> dist(0, 10);
    float offset = dist(mt);
    
    for(int i=0; i<N; i++)
    {
        A[i] = 0.0; //dist(mt);
        B[i] = 0.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    HIP_ASSERT( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
    HIP_ASSERT( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

    int thr_per_blk = 256;
    int blk_in_grid = 110;

    // Launch kernel
    hipLaunchKernelGGL(shared_atomic_add, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, offset);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    //hipError_t hipErrSync  = hipGetLastError();

    //// Check for errors on the GPU after control is returned to CPU
    //hipError_t hipErrAsync = hipDeviceSynchronize();

    //if (hipErrSync != hipSuccess) { printf("Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }
    //if (hipErrAsync != hipSuccess) { printf("Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

    // Copy data from device array d_C to host array C
    HIP_ASSERT( hipMemcpy(B, d_B, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    //for(int i=1; i<N; i++)
    //{
    //    if(fabs(B[i] - (A[i] + ITER*offset)) > 1.0) {
    //        printf("Error: a[%d]=%f, b[%d]=%f, offset = %f\n", i, A[i], i, B[i], offset);
    //        exit(-1);
    //    }

    //}
    // printf("offset = %f\n", offset);

    // Free CPU memory
    //free(A);
    //free(B);

    // Free GPU memory
    HIP_ASSERT( hipFree(d_A) );
    HIP_ASSERT( hipFree(d_B) );

//    printf("\n---------------------------\n");
//    printf("__SUCCESS__\n");
//    printf("---------------------------\n");

    return 0;
}