/*****************************************************************************
 Implementation of Fast Fourier Transformation on Finite Elements
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
#include <chrono>
#include <omp.h>
#include "fft_host.h"

typedef std::chrono::high_resolution_clock Clock;

#define LOG_NUM_THREADS 11 
#define NUM_THREADS 1 << LOG_NUM_THREADS
#define LOG_CONSTRAINTS 22 
#define CONSTRAINTS 1 << LOG_CONSTRAINTS

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

__device__ __forceinline__
size_t bitreverse(size_t n, const size_t l)
{
    size_t r = 0;
    for (size_t k = 0; k < l; ++k)
    {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

template<typename FieldT> 
__device__ __constant__ FieldT omega;
template<typename FieldT> 
__device__ __constant__ FieldT one;
template<typename FieldT> 
__device__ __constant__ FieldT zero;
template<typename FieldT>
__device__ FieldT field[CONSTRAINTS];
template<typename FieldT>
__device__ FieldT out[CONSTRAINTS];

template<typename FieldT>  __global__ void cuda_fft()
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t log_m = LOG_CONSTRAINTS;
    const size_t length = CONSTRAINTS;
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
    const size_t startidx = idx * block_length;
    assert (CONSTRAINTS == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a [block_length];
    memset(a, block_length,  zero<FieldT>); 
    //TODO change to zero element
    //TODO algorithm is non-deterministic because of padding

    FieldT omega_j = omega<FieldT>^idx;
    FieldT omega_step = omega<FieldT>^(idx<<(log_m - LOG_NUM_THREADS));
    
    FieldT elt = one<FieldT>;
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
        size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
        a[i] += field<FieldT>[id] * elt;
            elt *= omega_step;
        }
        elt *= omega_j;
    }

    FieldT omega_num_cpus = omega<FieldT> ^ NUM_THREADS;
    
    size_t n = block_length, logn = log2f(n);
    assert (n == (1u << logn));

    /* swapping in place (from Storer's book) */
    for (size_t k = 0; k < n; ++k)
    {
        const size_t rk = bitreverse(k, logn);
        if (k < rk)
        {
            FieldT tmp = a[k];
            a[k] = a[rk];
            a[rk] = tmp;
        }
    }

    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega_num_cpus^(n/(2*m));

        for (size_t k = 0; k < n; k += 2*m)
        {
            FieldT w = one<FieldT>;
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w * a[k+j+m];
                a[k+j+m] = a[k+j] - t;
                a[k+j] += t;
                w *= w_m;
            }
        }
        m *= 2;
    }
    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length)
        out<FieldT>[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
}

template<typename FieldT> void best_fft
    (std::vector<FieldT> &a, const FieldT &omg)
{
    FieldT* fld;
    CUDA_CALL (cudaGetSymbolAddress((void **)&fld, field<FieldT>));
    CUDA_CALL( cudaMemcpy(fld, &a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice);)
    
    const FieldT oneElem = 1; //FieldT::one
    const FieldT zeroElem = 0; //FieldT::zero
    cudaMemcpyToSymbol(omega<FieldT>, &omg, sizeof(FieldT), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(one<FieldT>, &oneElem, sizeof(FieldT), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(zero<FieldT>, &zeroElem, sizeof(FieldT), 0, cudaMemcpyHostToDevice);

    int blocks = NUM_THREADS/1024 + 1;
    int threads = NUM_THREADS > 1024 ? 1024 : NUM_THREADS; 
    printf("blocks %d, threads %d \n",blocks,threads);
    cuda_fft<FieldT> <<<blocks,threads>>>();
        
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    CUDA_CALL( cudaDeviceSynchronize();)
    
    FieldT* res;
    CUDA_CALL (cudaGetSymbolAddress((void**) &res, out<FieldT>));

    FieldT * result = (FieldT*) malloc (sizeof(FieldT) * a.size());    
    cudaMemcpy(result, fld, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost);

    a.assign(result, result + a.size());
    CUDA_CALL( cudaDeviceSynchronize();)
}


int main(void) 
{
    size_t size = CONSTRAINTS;
    int * array = (int*) malloc(size * sizeof(int));
    memset(array, 0x1234, size * sizeof(int));
    std::vector<int> v1(array, array+size);
    std::vector<int> v2 = v1;

    omp_set_num_threads( 8 );

    {
        {
            auto t1 = Clock::now();
            best_fft<int>(v1, 5678);
            auto t2 = Clock::now();
            printf("Device FFT took %ld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        {
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<int> (v2, 5678, LOG_NUM_THREADS, 1);
            auto t2 = Clock::now();
            printf("Host FFT took %ld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        //_basic_parallel_radix2_FFT_inner<int> (v1, 5678, 5, 1);
    }
    

    for(int j = 0; j < size; j++) {
        //printf("%d ", v1[j]);
    }
    printf("####################################\n");
    for(int j = 0; j < size; j++) {
  //        printf("%d ", v2[j]);
    }
    assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
