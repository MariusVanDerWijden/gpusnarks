#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>
#include "fft_host.h"
#include <cuda/fft_kernel.h>
#include <cuda/field.h>
//#include <fields/dummy_field.h>

typedef std::chrono::high_resolution_clock Clock;

int main(void) 
{
    size_t _size = 12;
    int * array = (int*) malloc(_size * sizeof(int));
    memset(array, 0x1234, _size * sizeof(int));
    std::vector<fields::Field> v1(array, array + _size);
    std::vector<fields::Field> v2 = v1;

    omp_set_num_threads( 8 );

    {
        {
            auto t1 = Clock::now();
            //void best_fft (FieldT *a, size_t _size, const FieldT &omg)
            best_fft<fields::Field>(&v1[0], v1.size(), fields::Field::to_field(123));
            auto t2 = Clock::now();
            printf("Device FFT took %ld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        {
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<fields::Field> (v2, fields::Field::to_field(123), 12, fields::Field::one());
            auto t2 = Clock::now();
            printf("Host FFT took %ld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        //_basic_parallel_radix2_FFT_inner<int> (v1, 5678, 5, 1);
    }
    

    for(int j = 0; j < _size; j++) {
        //printf("%d ", v1[j]);
    }
    printf("####################################\n");
    for(int j = 0; j < _size; j++) {
        //printf("%d ", v2[j]);
    }
    assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
