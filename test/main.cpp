#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>

// For multi-exp comment out the following define, and set the if(1) to if(0) in the CMakeList.txt.
//#define FFT

#ifdef FFT
#include "fft_host.h"
#include <cuda/fft_kernel.h>
#else
#include "multiexp.h"
#include <cuda/multi_exp.h>
#endif
#include <cuda/device_field.h>

typedef std::chrono::high_resolution_clock Clock;

void test_multiexp();
void test_fft();
void test_multiexp_mnt4753_G1();

int main(void)
{
    test_fft();
    test_multiexp();
    test_multiexp_mnt4753_G1();
}

void test_fft()
{
#ifdef FFT
    printf("\nTEST FFT\n");
    size_t _size = 1 << 16;
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
            auto t2 = Clock::now();
            printf("Device FFT took %ld \n",
                   std::chrono::duration_cast<
                       std::chrono::milliseconds>(t2 - t1)
                       .count());
        }

        {
            uint32_t _mod[SIZE] = {610172929, 1586521054, 752685471, 3818738770,
                                   2596546032, 1669861489, 1987204260, 1750781161, 3411246648, 3087994277,
                                   4061660573, 2971133814, 2707093405, 2580620505, 3902860685, 134068517,
                                   1821890675, 1589111033, 1536143341, 3086587728, 4007841197, 270700578, 764593169, 115910};
            auto t1 = Clock::now();
            _basic_parallel_radix2_FFT_inner<fields::Scalar>(v2, fields::Scalar(_mod), 10, fields::Scalar::one());
            auto t2 = Clock::now();
            printf("Host FFT took %ld \n",
                   std::chrono::duration_cast<
                       std::chrono::milliseconds>(t2 - t1)
                       .count());
        }
    }

    for (int i = 0; i < _size; i++)
    {
        fields::Scalar::testEquality(v1[i], v2[i]);
    }
    assert(v1 == v2);
    printf("\nDONE\n");
#endif
}

void test_multiexp()
{
#ifndef FFT
    printf("\nTEST MULTI_EXP\n");
    size_t _size = 1 << 18;
    std::vector<fields::Scalar> v1;
    std::vector<fields::Scalar> v2;
    std::vector<fields::Scalar> v3;
    std::vector<fields::Scalar> v4;

    v1.reserve(_size);
    v2.reserve(_size);
    v3.reserve(_size);
    v4.reserve(_size);

    for (size_t i = 0; i < _size; i++)
    {
        v1.push_back(fields::Scalar(1234));
        v2.push_back(fields::Scalar(1234));
        v3.push_back(fields::Scalar(1234));
        v4.push_back(fields::Scalar(1234));
    }

    fields::Scalar gpuResult;
    fields::Scalar cpuResult;

    {
        printf("Field size: %lu, Field count: %lu\n", sizeof(fields::Scalar), v1.size());
        auto t1 = Clock::now();
        gpuResult = multiexp<fields::Scalar, fields::Scalar>(v1, v2);
        auto t2 = Clock::now();
        printf("Device FFT took %ld \n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }

    {
        auto t1 = Clock::now();
        cpuResult = multi_exp<fields::Scalar, fields::Scalar>(v3, v4);
        auto t2 = Clock::now();
        printf("Host FFT took %ld \n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }

    fields::Scalar::testEquality(cpuResult, gpuResult);
    printf("\nDONE\n");
#endif
}

void test_multiexp_mnt4753_G1()
{
#ifndef FFT
    printf("\nTEST MULTI_EXP_MNT4753\n");
    size_t _size = 1 << 20;
    std::vector<fields::mnt4753_G1> v1;
    std::vector<fields::Scalar> v2;
    std::vector<fields::mnt4753_G1> v3;
    std::vector<fields::Scalar> v4;

    v1.reserve(_size);
    v2.reserve(_size);
    v3.reserve(_size);
    v4.reserve(_size);

    for (size_t i = 0; i < _size; i++)
    {
        v1.push_back(fields::mnt4753_G1(fields::Scalar(1234), fields::Scalar(1234), fields::Scalar(1234)));
        v2.push_back(fields::Scalar(1234));
        v3.push_back(fields::mnt4753_G1(fields::Scalar(1234), fields::Scalar(12345), fields::Scalar(123456)));
        v4.push_back(fields::Scalar(1234));
    }

    fields::mnt4753_G1 gpuResult;
    fields::mnt4753_G1 cpuResult;

    {
        printf("Field size: %lu, Field count: %lu\n", sizeof(fields::mnt4753_G1), v1.size());
        auto t1 = Clock::now();
        gpuResult = multiexp<fields::mnt4753_G1, fields::Scalar>(v1, v2);
        auto t2 = Clock::now();
        printf("Device FFT took %ld \n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }

    {
        auto t1 = Clock::now();
        cpuResult = multi_exp<fields::mnt4753_G1, fields::Scalar>(v3, v4);
        auto t2 = Clock::now();
        printf("Host FFT took %ld \n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }

    fields::mnt4753_G1::testEquality(cpuResult, gpuResult);
    printf("\nDONE\n");
#endif
}
