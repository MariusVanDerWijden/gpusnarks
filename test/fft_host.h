#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <math.h>

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

template <typename FieldT>
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
        FieldT w_m = omega;
        w_m = w_m ^ (n / (2 * m));

        asm volatile("/* pre-inner */");
        for (size_t k = 0; k < n; k += 2 * m)
        {
            FieldT w = one;
            for (size_t j = 0; j < m; ++j)
            {
                FieldT t = w;
                w = w * a[k + j + m];
                a[k + j + m] = a[k + j] - t;
                a[k + j] = a[k + j] + t;
                w = w * w_m;
            }
        }
        asm volatile("/* post-inner */");
        m *= 2;
    }
}

template <typename FieldT>
void _basic_parallel_radix2_FFT_inner(std::vector<FieldT> &a, const FieldT omega, const size_t log_cpus, const FieldT one)
{
    const size_t num_cpus = 1ul << log_cpus;

    const size_t m = a.size();
    const size_t log_m = log2(m);

    if (log_m < log_cpus)
    {
        _basic_serial_radix2_FFT(a, omega, one);
        return;
    }

    std::vector<std::vector<FieldT>> tmp(num_cpus);
    for (size_t j = 0; j < num_cpus; ++j)
    {
        tmp[j].resize(1ul << (log_m - log_cpus), FieldT::zero());
    }

#pragma omp parallel for
    for (size_t j = 0; j < num_cpus; ++j)
    {
        FieldT omega_j = omega;
        omega_j = omega_j ^ j;
        FieldT omega_step = omega;
        omega_step = omega_step ^ (j << (log_m - log_cpus));

        //printf("omega_host: %d %d \n", omega_j, omega_step);
        FieldT elt = FieldT::one();
        for (size_t i = 0; i < 1ul << (log_m - log_cpus); ++i)
        {
            for (size_t s = 0; s < num_cpus; ++s)
            {
                // invariant: elt is omega^(j*idx)
                const size_t idx = (i + (s << (log_m - log_cpus))) % (1u << log_m);
                FieldT temp = a[idx];
                temp = temp * elt;
                tmp[j][i] = tmp[j][i] + temp;
                //tmp[j][i] += a[idx] * elt;
                elt = elt * omega_step;
                //elt *= omega_step;
            }
            elt = elt * omega_j;
            //elt *= omega_j;
        }
    }
    FieldT omega_num_cpus = omega ^ num_cpus;
#pragma omp parallel for
    for (size_t j = 0; j < num_cpus; ++j)
    {
        _basic_serial_radix2_FFT(tmp[j], omega_num_cpus, one);
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_cpus; ++i)
    {
        for (size_t j = 0; j < 1ul << (log_m - log_cpus); ++j)
        {
            a[(j << log_cpus) + i] = tmp[i][j];
        }
    }
}