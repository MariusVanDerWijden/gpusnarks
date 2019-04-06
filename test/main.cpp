#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>
#include "fft_host.h"
#include <cuda/fft_kernel.h>
#include <cuda/device_field.h>
#include <fields/dummy_field.h>

typedef std::chrono::high_resolution_clock Clock;

int main(void) 
{
    size_t _size = 1 << 18;
    std::vector<fields::Field> v1;
    std::vector<dummy_fields::Field> v2;

    v1.resize(_size);
    v2.resize(_size);

    for(size_t i = 0; i < _size; i++)
    {
    	v1.push_back(fields::Field(1234));
    	v2.push_back(dummy_fields::Field(1234));
    }

    omp_set_num_threads( 8 );

    {
        {
        	printf("Field size: %d, Field count: %d\n", sizeof(fields::Field),v1.size());
        	printf("A address: %p Last element: %d\n", &v1[0], v1[_size -1]);
            auto t1 = Clock::now();
            best_fft<fields::Field>(v1, v1.size(), fields::Field(123));
            //best_fft<fields::Field>(&v1[0], v1.size(), fields::Field(123));
            auto t2 = Clock::now();
            printf("Device FFT took %ld \n",
                std::chrono::duration_cast<
                std::chrono::milliseconds>(t2 - t1).count());
        }
        
        {
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<dummy_fields::Field> (v2, dummy_fields::Field(123), 12, dummy_fields::Field::one());
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
    //assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
