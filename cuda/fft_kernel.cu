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
#include "fft_kernel.h"
#include "device_field.h"
#include "device_field_operators.h"

#define LOG_NUM_THREADS 16
#define NUM_THREADS (1 << LOG_NUM_THREADS)
#define LOG_CONSTRAINTS 18
#define CONSTRAINTS (1 << LOG_CONSTRAINTS)

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

__device__ uint32_t _mod [SIZE] = { 610172929, 1586521054, 752685471, 3818738770, 
    2596546032, 1669861489, 1987204260, 1750781161, 3411246648, 3087994277, 
    4061660573, 2971133814, 2707093405, 2580620505, 3902860685, 134068517, 
    1821890675, 1589111033, 1536143341, 3086587728, 4007841197, 270700578, 764593169, 115910};

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
    //zero()
    memset(a, block_length,  0); 

    //TODO algorithm is non-deterministic because of padding
    FieldT omega_j = FieldT(_mod);
    omega_j = omega_j ^ idx; // pow
    FieldT omega_step = omega<FieldT>;
    omega_step = omega_step ^ idx << (log_m - LOG_NUM_THREADS);
    
    FieldT elt = one<FieldT>;
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
            size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
            FieldT tmp = field<FieldT>[id];
            tmp = tmp * elt;
            a[i] = a[i] + tmp;
            elt = elt * omega_step;
        }
        elt = elt * omega_j;
    }

    const FieldT omega_num_cpus = omega<FieldT> ^ NUM_THREADS;
    
    //Do not remove log2f(n), otherwise register overflow
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
                a[k+j] = a[k+j] + t;
                w = w * w_m;
            }
        }
        m = m << 1;
    }
    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length)
            out<FieldT>[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
}

template<typename FieldT> 
void best_fft (std::vector<FieldT> &a, const FieldT &omg)
{
	int cnt;
    cudaGetDeviceCount(&cnt);
    printf("CUDA Devices: %d, Field size: %d, Field count: %d\n", cnt, sizeof(FieldT), a.size());
    assert(a.size() == CONSTRAINTS);

    CUDA_CALL( cudaMemcpyToSymbol(field<FieldT>, &a[0], CONSTRAINTS, cudaMemcpyHostToDevice) );
    
    const FieldT oneElem = FieldT::one();
    const FieldT zeroElem = FieldT::zero();
    CUDA_CALL( cudaMemcpyToSymbol(omega<FieldT>, &omg, sizeof(FieldT), 0, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpyToSymbol(one<FieldT>, &oneElem, sizeof(FieldT), 0, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpyToSymbol(zero<FieldT>, &zeroElem, sizeof(FieldT), 0, cudaMemcpyHostToDevice) );

    size_t blocks = NUM_THREADS / 1024 + 1;
    size_t threads = NUM_THREADS > 1024 ? 1024 : NUM_THREADS;
    printf("threads %d, blocks %d, threads %d \n",NUM_THREADS, blocks, threads);
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
    cudaMemcpy(result, res, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost);

    //std::copy(result, result + _size, a);
    CUDA_CALL( cudaDeviceSynchronize();)
}

//List with all templates that should be generated
//template void best_fft(std::vector<int> &a, const int &omg);
//template void best_fft(fields::Field *v, size_t _size, const fields::Field &omg);
template void best_fft(std::vector<fields::Scalar> &v, const fields::Scalar &omg);
