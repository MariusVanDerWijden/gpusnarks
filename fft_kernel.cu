#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
typedef std::chrono::high_resolution_clock Clock;


#define LOG_NUM_THREADS 3
#define NUM_THREADS 8
#define MULTICORE true
#define CONSTRAINTS 4194304
#define LOG_CONSTRAINTS 22

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

size_t bitreverse_host(size_t n, const size_t l)
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

template<typename FieldT>  __global__ void cuda_fft(
    FieldT *field, size_t const length)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d ",omega<FieldT>);
    const size_t log_m = LOG_CONSTRAINTS;
    
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
    const size_t startidx = idx * block_length;
    assert (length == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a[block_length];
	    //= (FieldT*)malloc(block_length * sizeof(FieldT));
    memset(a, block_length,  0); //TODO change to zero element
    FieldT omega_j = omega<FieldT>^idx;
    FieldT omega_step = omega<FieldT>^(idx<<(log_m - LOG_NUM_THREADS));
    
    FieldT elt = one<FieldT>;
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
            //size_t mod = (1u << log_m); //mod guaranteed to be 2^n
	        size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
	        if(id > length)
	            continue;
            //size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) & (mod - 1);
	    a[i] += field[id] * elt;
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
            // FieldT * w = (FieldT *) malloc (sizeof(FieldT));
            // memcpy(w, one, sizeof(FieldT));
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
        // now: i = idx >> (log_m - log_cpus) and j = idx % (1u << (log_m - log_cpus)), for idx = ((i<<(log_m-log_cpus))+j) % (1u << log_m)
        if(((j << LOG_NUM_THREADS) + idx) < length)
	       field[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
    free(a);
}

template<typename FieldT> void best_fft
    (std::vector<FieldT> &a, const FieldT &omg, const FieldT &oneElem)
    {
        FieldT * array;
        CUDA_CALL( cudaMalloc((void**) &array, sizeof(FieldT) * a.size());)
        CUDA_CALL( cudaMemcpy(array, &a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice);)
        

        CUDA_CALL (cudaMemcpyToSymbol(omega<FieldT>, &omg, sizeof(FieldT), 0, cudaMemcpyHostToDevice);)
        CUDA_CALL (cudaMemcpyToSymbol(one<FieldT>, &oneElem, sizeof(FieldT), 0, cudaMemcpyHostToDevice);)

	    int blocks = NUM_THREADS/1024 + 1;
	    int threads = NUM_THREADS > 1024 ? 1024 : NUM_THREADS; 
        cuda_fft<FieldT><<<blocks,threads>>>(array, a.size());
        CUDA_CALL( cudaDeviceSynchronize();)

        FieldT * result = (FieldT*) malloc (sizeof(FieldT) * a.size());	
        cudaMemcpy(result, array, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost);
        a.assign(result, result + a.size());
        CUDA_CALL( cudaDeviceSynchronize();)
 //       printf("%d tick", result[3]);
    }

template<typename FieldT>
void _basic_serial_radix2_FFT(std::vector<FieldT> &a, const FieldT omega, const FieldT one)
{
    const size_t n = a.size(), logn = log2(n);
    
    /* swapping in place (from Storer's book) */
    for (size_t k = 0; k < n; ++k)
    {
        const size_t rk = bitreverse_host(k, logn);
        if (k < rk)
            std::swap(a[k], a[rk]);
    }

    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega^(n/(2*m));

        asm volatile  ("/* pre-inner */");
        for (size_t k = 0; k < n; k += 2*m)
        {
            FieldT w = one;
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w * a[k+j+m];
                a[k+j+m] = a[k+j] - t;
                a[k+j] += t;
                w *= w_m;
            }
        }
        asm volatile ("/* post-inner */");
        m *= 2;
    }
}

template<typename FieldT>
void _basic_parallel_radix2_FFT_inner(std::vector<FieldT> &a, const FieldT omega, const size_t log_cpus, const FieldT one)
{
    const size_t num_cpus = 1ul<<log_cpus;

    const size_t m = a.size();
    const size_t log_m = log2(m);
   
    if (log_m < log_cpus)
    {
        _basic_serial_radix2_FFT(a, omega, one);
        return;
    }

    std::vector<std::vector<FieldT> > tmp(num_cpus);
    for (size_t j = 0; j < num_cpus; ++j)
    {
        tmp[j].resize(1ul<<(log_m-log_cpus), 0);
    }

    #pragma omp parallel for
    for (size_t j = 0; j < num_cpus; ++j)
    {
        const FieldT omega_j = omega^j;
        const FieldT omega_step = omega^(j<<(log_m - log_cpus));
	
	    //printf("omega_host: %d %d \n", omega_j, omega_step);
        FieldT elt = one;
        for (size_t i = 0; i < 1ul<<(log_m - log_cpus); ++i)
        {
            for (size_t s = 0; s < num_cpus; ++s)
            {
                // invariant: elt is omega^(j*idx)
                const size_t idx = (i + (s<<(log_m - log_cpus))) % (1u << log_m);
                tmp[j][i] += a[idx] * elt;
                elt *= omega_step;
            }
            elt *= omega_j;
        }
    }
    printf("host: %d \n ", tmp[0][0]);
    const FieldT omega_num_cpus = omega^num_cpus;

    #pragma omp parallel for
    for (size_t j = 0; j < num_cpus; ++j)
    {
        _basic_serial_radix2_FFT(tmp[j], omega_num_cpus, one);
    }


    #pragma omp parallel for
    for (size_t i = 0; i < num_cpus; ++i)
    {
        for (size_t j = 0; j < 1ul<<(log_m - log_cpus); ++j)
        {
            // now: i = idx >> (log_m - log_cpus) and j = idx % (1u << (log_m - log_cpus)), for idx = ((i<<(log_m-log_cpus))+j) % (1u << log_m)
            a[(j<<log_cpus) + i] = tmp[i][j];
        }
    }
}

int main(void) {

   // size_t size = 268435456;
    size_t size = CONSTRAINTS;
    //size_t size = 65536;
    int * array = (int*) malloc(size * sizeof(int));
    memset(array, 0x1234, size * sizeof(int));
    std::vector<int> v1(array, array+size);
    std::vector<int> v2 = v1;

   // printf("max_threads: %d \n", omp_get_max_threads());
    omp_set_num_threads( 8 );

    {
        {
            auto t1 = Clock::now();
            best_fft<int>(v1, 5678, 1);
            auto t2 = Clock::now();
            printf("Device FFT took %lld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        {
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<int> (v2, 5678, LOG_NUM_THREADS, 1);
            auto t2 = Clock::now();
            printf("Host FFT took %lld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        
       // _basic_parallel_radix2_FFT_inner<int> (v1, 5678, 5, 1);
    }
    

    for(int j = 0; j < size; j++) {
        //printf("%d ", v1[j]);
    }
    printf("####################################\n");
    for(int j = 0; j < size; j++) {
  //	    printf("%d ", v2[j]);
    }
    assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
