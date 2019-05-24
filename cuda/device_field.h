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

#ifndef DEBUG
#include <cuda.h>
#include <cuda_runtime.h>

#define cu_fun __host__ __device__ 
#else

#define cu_fun
#include <cstdio>
#include <cassert>

#endif

#define SIZE (256 / 32)



namespace fields{

using size_t = decltype(sizeof 1ll);

#ifndef DEBUG
__constant__
#endif
uint32_t _mod [SIZE];
//leading zeros of mod
uint32_t _mod_lz;

struct Field {
	//Intermediate representation
	uint32_t im_rep [SIZE] = {0};
    //Returns zero element
    cu_fun static Field zero()
    {
        Field res;
        for(size_t i = 0; i < SIZE; i++)
            res.im_rep[i] = 0;
        return res;
    }

    //Returns one element
    cu_fun static Field one()
    {
        Field res;
            res.im_rep[SIZE - 1] = 1;
        return res;
    }
    //Default constructor
    Field() = default;
    //Construct from value
    cu_fun Field(const uint32_t value)
    {
        im_rep[SIZE - 1] = value;
    }

    cu_fun Field(const uint32_t* value)
    {
        for(size_t i = 0; i < SIZE; i++)
            im_rep[i] = value[i];
    }
};

#ifdef DEBUG
    void printField(fields::Field f)
    {
        for(size_t i = 0; i < SIZE; i++)
            printf("%u, ", f.im_rep[i]);
        printf("\n");
    }

    void testEquality(fields::Field f1, fields::Field f2)
    {
        for(size_t i = 0; i < SIZE; i++)
            if(f1.im_rep[i] != f2.im_rep[i])
            {
                printField(f1);
                printField(f2);
                assert(!"missmatch");
            }
    }
#endif

}