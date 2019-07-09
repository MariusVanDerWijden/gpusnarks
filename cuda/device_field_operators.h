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

#pragma once
#include <cstdint>

#include "device_field.h"

#define SIZE (768 / 32)

#if defined(__CUDA_ARCH__)
#define _clz __clz
#else
#define _clz __builtin_clz
#endif

#ifndef DEBUG
#define cu_fun __host__ __device__ 
#else
#define cu_fun
#include <assert.h>
#include <malloc.h>
#include <cstring>
#endif

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))

namespace fields{

using size_t = decltype(sizeof 1ll);

cu_fun bool operator==(const Scalar& lhs, const Scalar& rhs)
{
    for(size_t i = 0; i < SIZE; i++)
        if(lhs.im_rep[i] != rhs.im_rep[i])
            return false;
    return true;
}

cu_fun uint32_t clz(const uint32_t* element, const size_t e_size)
{
    uint32_t lz = 0;
    uint32_t tmp;
    for(size_t i = 0; i < e_size; i++)
    {
        if(element[i] == 0)
            tmp = 32;
        else
            tmp = _clz(element[i]);
        lz += tmp;
        if(tmp < 32)
            break;
    }
    return lz;
}

cu_fun long idxOfLNZ(const Scalar& fld)
{
    return SIZE - clz(fld.im_rep, SIZE);
}

cu_fun bool hasBitAt(const Scalar& fld, long index)
{
    long idx1 = index % SIZE;
    long idx2 = index / SIZE;
    return CHECK_BIT(fld.im_rep[idx2], idx1) != 0;  
}

#ifdef DEBUG
cu_fun void set_mod(const Scalar& f)
{
    for(size_t i = 0; i < SIZE; i++)
        _mod[i] = f.im_rep[i];
}
#endif

//Returns true if the first element is less than the second element
cu_fun bool less(const uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size == e2_size);
    for(size_t i = 0; i < SIZE; i++)
        if(element1[i] > element2[i])
            return false;
        else if(element1[i] < element2[i])
            return true;
    return false;
}

// Returns the carry, true if there was a carry, false otherwise
cu_fun bool _add(uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size == e2_size);
    bool carry = false;
    for(size_t i = 1; i <= e1_size; i++)
    {
        uint64_t tmp = (uint64_t)element1[e1_size - i];
        if(carry) tmp++;
        element1[e1_size - i] = tmp + (uint64_t)element2[e1_size - i];
        carry = (tmp >> 32) > 0;
    }
    return carry;
}

// Fails if the second number is bigger than the first
cu_fun void _subtract(uint32_t* element1, const size_t e1_size, bool carry, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size == e2_size);
    bool borrow = false;
    for(size_t i = 1; i <= e1_size; i++)
    {
        uint64_t tmp = (uint64_t)element1[e1_size - i];
        if(borrow) tmp--;
        borrow = (tmp < element2[e2_size - i]);
        if(borrow) tmp += ((uint64_t)1 << 33);
        element1[e1_size - i] = tmp - element2[e2_size - i];
    }
    assert(carry == borrow);
}

cu_fun void ciosMontgomeryMultiply(uint32_t * result, 
const uint32_t* a, const size_t a_size, 
const uint32_t* b, const uint32_t* n,
const uint64_t m_prime)
{
    uint64_t temp;
    for(size_t i = 0; i < a_size; i++)
    {
        uint32_t carry = 0;
        for(size_t j = 0; j < a_size; j++)
        {
            temp = result[j];
            temp += (uint64_t)a[j] * (uint64_t)b[i];
            temp += carry;
            result[j] = (uint32_t)temp;
            carry = temp >> 32;
        }
        temp = result[a_size] + carry;
        result[a_size] = (uint32_t) temp;
        result[a_size + 1] = temp >> 32;
        uint64_t m = (result[0] * m_prime) & 4294967296;
        temp = result[0] + (uint64_t)m * n[0]; 
        carry = temp >> 32;
        for(size_t j = 0; j < a_size; j++)
        {
            temp = result[j];
            temp += (uint64_t)m * (uint64_t)n[j];
            temp += carry;
            result[j - 1] = (uint32_t)temp;
            carry = temp >> 32;
        }
        temp = result[a_size] + carry;
        result[a_size - 1] = (uint32_t) temp;
        result[a_size] = result[a_size + 1] + temp >> 32;
    }
    uint32_t t[SIZE];
    memcpy(t, result, a_size);
    int64_t stemp = 0;
    int carry = 0;
    for(size_t i = 0; i < a_size; i++)
    {
        stemp = (int64_t)result[i];
        stemp -= (int64_t)n[i];
        stemp -= carry;
        if(stemp < 0)
        {
            t[i] = (uint32_t) (stemp + 4294967296);
            carry = 1;
        }
        else
        {
            t[i] = stemp;
            carry = 0;
        }
    }
    stemp = t[a_size] - carry;
    t[a_size] = (uint32_t)stemp;
    if(stemp < 0)
        memcpy(result, t, a_size);
}

//Adds two elements
cu_fun void Scalar::add(Scalar & fld1, const Scalar & fld2) const
{
    bool carry = _add(fld1.im_rep, SIZE, fld2.im_rep, SIZE);
    if(carry || less(_mod, SIZE, fld1.im_rep, SIZE))
        _subtract(fld1.im_rep, SIZE, false, _mod, SIZE);
}

//Subtract element two from element one
cu_fun void Scalar::subtract(Scalar & fld1, const Scalar & fld2) const
{
    bool carry = false;
    if(less(fld1.im_rep, SIZE, fld2.im_rep, SIZE))
        carry = _add(fld1.im_rep, SIZE, _mod, SIZE);
    _subtract(fld1.im_rep, SIZE, carry, fld2.im_rep, SIZE);
}

//Multiply two elements
cu_fun void Scalar::mul(Scalar & fld1, const Scalar & fld2) const
{
    uint32_t tmp[SIZE * 2];
    memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));
    
    ciosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, fld2.im_rep, _mod, 4294967296L);
    for(size_t i = 0; i < SIZE; i++)
        fld1.im_rep[i] = tmp[i];
}

//Exponentiates this element
cu_fun void Scalar::pow(Scalar & fld1, const size_t pow)
{
    if(pow == 0) 
    {
        fld1 = Scalar::one();
        return;
    }

    if(pow == 1)
    {
        return;
    }

    uint32_t tmp[SIZE * 2];
    uint32_t temp[SIZE];

    for(size_t i = 0; i < SIZE; i++)
        temp[i] = fld1.im_rep[i];

    for(size_t i = 0; i < pow - 1; i++)
    {
        memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));
        
        ciosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, temp, _mod, 4294967296L);
        for(size_t i = 0; i < SIZE; i++)
            fld1.im_rep[i] = tmp[i];
    }
}

}
