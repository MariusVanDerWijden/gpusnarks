#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>
#include "fft_host.h"
#include <cuda/fft_kernel.h>

typedef std::chrono::high_resolution_clock Clock;

int main(void) 
{
    size_t size = 12;
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
            _basic_parallel_radix2_FFT_inner<int> (v2, 5678, 12, 1);
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
        //printf("%d ", v2[j]);
    }
    assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
