#include <assert.h>
#include <random>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define N 110*256
#define ITER 102400

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

int main()
{
    size_t bytes = N*sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);

    float *d_A, *d_B;
    HIP_ASSERT( hipMalloc(&d_A, bytes) );
    HIP_ASSERT( hipMalloc(&d_B, bytes) );

    std::random_device rand;
    std::mt19937 mt(rand());
    std::uniform_real_distribution<> dist(0, 10);
    float offset = dist(mt);
    
    for(int i=0; i<N; i++)
    {
        A[i] = 0.0; //dist(mt);
        B[i] = 0.0;
    }

    HIP_ASSERT( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
    HIP_ASSERT( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

    // This is sized for MI250X - 110 CUs, 4 waves per WG x 64 threads per wave.
    int thr_per_blk = 256;
    int blk_in_grid = 110;

    hipLaunchKernelGGL(shared_atomic_add, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, offset);

    //hipError_t hipErrSync  = hipGetLastError();

    //hipError_t hipErrAsync = hipDeviceSynchronize();

    //if (hipErrSync != hipSuccess) { printf("Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }
    //if (hipErrAsync != hipSuccess) { printf("Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

    HIP_ASSERT( hipMemcpy(B, d_B, bytes, hipMemcpyDeviceToHost) );

    // Check results
    //for(int i=1; i<N; i++)
    //{
    //    if(fabs(B[i] - (A[i] + ITER*offset)) > 1.0) {
    //        printf("Error: a[%d]=%f, b[%d]=%f, offset = %f\n", i, A[i], i, B[i], offset);
    //        exit(-1);
    //    }

    //}
    // printf("offset = %f\n", offset);

    //free(A);
    //free(B);

    HIP_ASSERT( hipFree(d_A) );
    HIP_ASSERT( hipFree(d_B) );

//    printf("Done!\n");

    return 0;
}