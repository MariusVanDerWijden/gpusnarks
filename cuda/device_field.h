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

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE (256 / 32)

#define cu_fun __host__ __device__ 

namespace fields{

using size_t = decltype(sizeof 1ll);

__constant__
uint32_t _mod [SIZE];

struct Field {
	//Intermediate representation
	uint32_t im_rep [SIZE];
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
    cu_fun Field(uint32_t value)
    {
        im_rep[SIZE - 1] = value;
    }

    cu_fun Field(uint32_t value[])
    {
        for(size_t i = 0; i < SIZE; i++)
            im_rep[i] = value[i];
    }
};

}