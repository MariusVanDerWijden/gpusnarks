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
//#define m_inv 1723983939ULL
#define m_inv 4294967296L
//#define m_inv 1L
//#define m_inv 85162L

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
    uint32_t carry = 0;
    for(int i = e1_size - 1; i >= 0; i--)
    {
        uint64_t tmp = (uint64_t)element1[i];
        tmp += carry;
        tmp += (uint64_t)element2[i];
        element1[i] = (uint32_t) (tmp);
        carry = (uint32_t) (tmp >> 32);
    }
    return carry;
}

// Fails if the second number is bigger than the first
cu_fun bool _subtract(uint32_t* element1, const size_t e1_size, bool carry, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size == e2_size);
    bool borrow = false;
    for(int i = e1_size - 1; i >= 0; i--)
    {
        uint64_t tmp = (uint64_t)element1[i];
        bool underflow = (tmp == 0) && (element2[i] > 0 || borrow);
        if(borrow) tmp--;
        borrow = underflow || (tmp < element2[i]);
        if(borrow) tmp += ((uint64_t)1 << 33);
        element1[i] = tmp - element2[i];
    }
    //assert(borrow == carry);
    return borrow;
}

cu_fun void montyNormalize(uint32_t * result, const size_t a_size, const bool msb) 
{
    uint32_t u[SIZE + 1] = {0};
    memcpy(u, result, a_size + 1);
    bool borrow = _subtract(u, SIZE, false, _mod, SIZE);
    if(msb || !borrow) 
    {
        assert(!msb || msb == borrow);
        memcpy(result, u, a_size + 1);
    }
}

cu_fun void ciosMontgomeryMultiply(uint32_t * result, 
const uint32_t* a, const size_t a_size, 
const uint32_t* b, const uint32_t* n)
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
        temp = result[a_size - 1] + carry;
        result[a_size - 1] = (uint32_t) temp;
        result[a_size] = temp >> 32;
        uint64_t m = ((uint64_t)result[0] * m_inv) % 4294967296;
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
        temp = result[a_size -1] + carry;
        result[a_size - 2] = (uint32_t) temp;
        result[a_size - 1] = result[a_size] + temp >> 32;
    }
    bool msb = false;
    montyNormalize(result, a_size, msb);
}

cu_fun void sosMontgomeryMultiply(uint32_t * result, 
const uint32_t* a, const size_t a_size, 
const uint32_t* b, const uint32_t* n)
{
    uint64_t temp;
    for(int i = a_size - 1; i >= 0; i--)
    {
        uint64_t carry = 0;
        for(int j = a_size - 1; j >= 0; j--)
        {
            temp = result[i + j];
            temp += (uint64_t)a[j] * (uint64_t)b[i];
            temp += carry;
            result[i + j] = (uint32_t)temp;
            carry = temp >> 32;
        }
        result[a_size - i - 1] = carry;
    }

    for(int i = a_size - 1; i >= 0; i--)
    {
        uint64_t carry = 0;
        uint64_t m = ((uint64_t)result[i] * m_inv) % 1<<32;  
        for(int j = a_size - 1; j >= 0; j--)
        {
            temp = result[i + j];
            temp += m * (uint64_t)n[j];
            temp += carry;
            result[i + j] = (uint32_t)temp;
            carry = temp >> 32;
        }
        //TODO ADD() propagates the carry upwards
        uint64_t tmp = result[a_size - i - 1];
        tmp += carry;
        result[a_size - i - 1] = tmp;
        assert(tmp >> 32 == 0);
    }
     
    uint32_t u[SIZE] = {0};
    for (int i = 0; i < SIZE; i++) 
    {
        u[i] = result[a_size + i - 1]; 
    }
    temp = 0;
    uint64_t carry = 0;
    for(int i = a_size - 1; i >= 0; i--)
    {
        temp = u[i];
        temp -= (uint64_t)n[i];
        temp -= carry;
        result[i] = (uint32_t)temp;
        carry = temp >> 32;
    }
    temp = u[0];
    temp -= carry;
    result[a_size - 1] = (uint32_t)temp;
    carry = temp >> 32;
    if (carry != 0)
    {
        memcpy(result + a_size - 1, u, a_size);
    }
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
    //printScalar(fld1);
    //printScalar(fld2);
    if(less(fld1.im_rep, SIZE, fld2.im_rep, SIZE))
        carry = _add(fld1.im_rep, SIZE, _mod, SIZE);
    //printScalar(fld1);
    _subtract(fld1.im_rep, SIZE, carry, fld2.im_rep, SIZE);
    //printScalar(fld1);
}

//Multiply two elements
cu_fun void Scalar::mul(Scalar & fld1, const Scalar & fld2) const
{
    uint32_t tmp[SIZE*2 + 2] = {0};
    
    //ciosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, fld2.im_rep, _mod);
    sosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, fld2.im_rep, _mod);
    for(size_t i = 0; i < SIZE; i++)
        fld1.im_rep[i] = tmp[i + SIZE];
    printScalar(Scalar(fld1));     
    //printScalar(Scalar(tmp));   
}

cu_fun void to_monty(Scalar & a) {
    // a = a << 2^(32*SIZE)
    // a = a % _mod
}

cu_fun void from_monty(Scalar & a) {
    Scalar s = Scalar::one();
    a = a * s;
}

//Exponentiates this element
cu_fun void Scalar::pow(Scalar & fld1, const uint32_t pow) const
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

    uint32_t tmp[SIZE * 2 + 1];
    uint32_t temp[SIZE];

    to_monty(fld1);

    for(size_t i = 0; i < SIZE; i++)
        temp[i] = fld1.im_rep[i];

    for(size_t i = 0; i < pow - 1; i++)
    {
        memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));
        sosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, temp, _mod);
        for(size_t i = 0; i < SIZE; i++)
            fld1.im_rep[i] = tmp[i + SIZE];
        //printScalar(Scalar(fld1));
        // Do not delete this, otherwise invalid compiler optimization
        //printScalar(Scalar(temp));
        //printScalar(Scalar(tmp));

    }
    from_monty(fld1);
}

}
