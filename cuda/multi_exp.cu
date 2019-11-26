/*****************************************************************************
 Implementation of Multi-Exponentiation on Finite Elements
 *****************************************************************************
 * @author     Marius van der Wijden
 * Copyright [2019] [Marius van der Wijden]
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>

#include "multi_exp.h"
#include "device_field.h"
#include "device_field_operators.h"

#define LOG_NUM_THREADS 16
#define NUM_THREADS (1 << LOG_NUM_THREADS)

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

template <typename FieldT>
__inline__ __device__
FieldT warpReduceSum(FieldT x) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        FieldT y = FieldT::shuffle_down(__activemask(), x, offset);
        x = x + y;
    }
    return x;
}

template <typename FieldT>
extern __shared__ FieldT sMem[];

template <typename FieldT>
__inline__ __device__
FieldT blockReduceSum(FieldT x) {
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    x = warpReduceSum<FieldT>(x); 
    if (lane==0) sMem<FieldT>[warpId]=x;
    __syncthreads();
    if(threadIdx.x < blockDim.x / warpSize)
        x = sMem<FieldT>[lane];
    else 
        x = FieldT::zero();
    if (warpId==0) x = warpReduceSum<FieldT>(x);
    return x;
}

template <typename FieldT>
__global__ void
deviceReduceKernelSecond(FieldT *out, const FieldT *resIn, const size_t n) { 
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    FieldT sum = FieldT::zero();
    if (idx < n) {
        for(int i = idx; i < n; i += (blockDim.x * gridDim.x)){
            sum = sum + resIn[i];
        }
        sum = blockReduceSum<FieldT>(sum);
    }   
    if (threadIdx.x==0) // Store the end result
        out[blockIdx.x] = sum; 
}

template <typename FieldT, typename FieldMul>
__global__ void 
deviceReduceKernel(FieldT *result, const FieldT *a, const FieldMul *mul, const size_t n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    FieldT sum = FieldT::zero();
    if (idx < n) {
        for(int i = idx; i < n; i += (blockDim.x * gridDim.x)){
            const FieldT tmp = a[i] * mul[i];
            sum = sum + tmp;   
        }
        sum = blockReduceSum<FieldT>(sum);
    }
    if (threadIdx.x==0)
        result[blockIdx.x] = sum;
}

// Multiexp is a function that performs a multiplication and a summation of all elements.
template<typename FieldT, typename FieldMul> 
FieldT multiexp (std::vector<FieldT> &a, std::vector<FieldMul> &mul) {
    assert(a.size() == mul.size());
    size_t threads = sizeof(FieldT) == 96 ? 256 : 128;
    size_t blocks = min((a.size() + threads - 1) / threads, (unsigned long)256);
    size_t sMem = threads * sizeof(FieldT);
    
    FieldT *in;
    CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * a.size()); )
    CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice); )

    FieldMul *cmul;
    CUDA_CALL( cudaMalloc((void**)&cmul, sizeof(FieldMul) * a.size()); )
    CUDA_CALL( cudaMemcpy(cmul, (void**)&mul[0], sizeof(FieldMul) * a.size(), cudaMemcpyHostToDevice); )

    FieldT *temp;
    CUDA_CALL( cudaMalloc((void**)&temp, sizeof(FieldT) * blocks); )
    deviceReduceKernel<FieldT,FieldMul><<<blocks, threads, sMem>>>(temp, in, cmul, a.size());
    
    FieldT *out;
    CUDA_CALL( cudaMalloc(&out, sizeof(FieldT)); )
    deviceReduceKernelSecond<FieldT><<<1, 256, sMem>>>(out, temp, blocks);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-2);
    }

    FieldT cpuResult;
    cudaMemcpy((void**)&cpuResult, out, sizeof(FieldT), cudaMemcpyDeviceToHost);
    return cpuResult;
}

// Make templates for the most common types.
template fields::Scalar multiexp(std::vector<fields::Scalar> &v, std::vector<fields::Scalar> &mul);
template fields::fp2 multiexp(std::vector<fields::fp2> &v, std::vector<fields::Scalar> &mul);
template fields::mnt4753_G1 multiexp(std::vector<fields::mnt4753_G1> &v, std::vector<fields::Scalar> &mul);