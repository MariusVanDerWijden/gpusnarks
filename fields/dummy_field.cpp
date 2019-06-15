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
#include "dummy_field.h"
#include "assert.h"

#define size (256 / 32)

namespace dummy_fields{

uint32_t Field::mod = 0;

//Returns zero element
Field Field::zero()
{
    Field f;
    f.im_rep = 0;
    return f;
}

//Returns one element
Field Field::one()
{
    Field f;
    f.im_rep = 1;
    return f;
}

//Returns true iff this element is zero
bool Field::is_zero(const Field & fld)
{
    return fld.im_rep == 0;
}

//Squares this element
void Field::square(Field & fld)
{
    fld.im_rep = fld.im_rep * fld.im_rep;
}

/*
//Doubles this element
void Field::double(Field & fld)
{
    uint32_t temp[] = {2};
    uint32_t tmp[] = multiply(fld.im_rep, size, temp, 1);
    //size of tmp is 2*size
    modulo(tmp, 2*size, mod, size);
    //Last size words are the result
    for(size_t i = 0; i < size; i++)
        fld.im_rep[i] = tmp[size + i]; 
}*/

//Negates this element
void Field::negate(Field & fld)
{
    //TODO implement
}

//Adds two elements
void Field::add(Field & fld1, const Field & fld2)
{
    fld1.im_rep += fld2.im_rep;
}

//Subtract element two from element one
void Field::subtract(Field & fld1, const Field & fld2)
{
    fld1.im_rep -= fld2.im_rep;
}

//Multiply two elements
void Field::mul(Field & fld1, const Field & fld2)
{
    fld1.im_rep *= fld2.im_rep;
}

//Computes the multiplicative inverse of this element, if nonzero
void Field::mul_inv(Field & fld1)
{
    //TODO implement
}

//Exponentiates this element
void Field::pow(Field & fld1, const size_t pow)
{
    uint32_t tmp = fld1.im_rep;
    for(size_t i = 0; i < pow; i++)
    {
        tmp *= fld1.im_rep;
    }
}

}