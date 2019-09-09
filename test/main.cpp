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
#include <cuda/multi_exp.h>
#include <fields/dummy_field.h>

typedef std::chrono::high_resolution_clock Clock;

int main(void)
{
    size_t _size = 1 << 20;
    std::vector<fields::Scalar> v1;
    std::vector<fields::Scalar> v2;

    v1.reserve(_size);
    v2.reserve(_size);

    for (size_t i = 0; i < _size; i++)
    {
        v1.push_back(fields::Scalar(1234));
        v2.push_back(fields::Scalar(1234));
    }

    omp_set_num_threads(8);

    {
        {
            printf("Field size: %lu, Field count: %lu\n", sizeof(fields::Scalar), v1.size());
            auto t1 = Clock::now();
            best_fft<fields::Scalar>(v1, fields::Scalar(123));
            //best_fft<fields::Scalar>(&v1[0], v1.size(), fields::Scalar(123));
            auto t2 = Clock::now();
            printf("Device FFT took %ld \n",
                   std::chrono::duration_cast<
                       std::chrono::milliseconds>(t2 - t1)
                       .count());
        }

        {
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<fields::Scalar>(v2, fields::Scalar(123), 12, fields::Scalar::one());
            auto t2 = Clock::now();
            printf("Host FFT took %ld \n",
                   std::chrono::duration_cast<
                       std::chrono::milliseconds>(t2 - t1)
                       .count());
        }

        //_basic_parallel_radix2_FFT_inner<int> (v1, 5678, 5, 1);
    }

    for (int j = 0; j < _size; j++)
    {
        //printf("%d ", v1[j]);
    }
    printf("####################################\n");
    for (int j = 0; j < _size; j++)
    {
        //printf("%d ", v2[j]);
    }
    //assert(v1 == v2);
    printf("\nDONE\n");
    return 0;
}
