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

namespace fields{

//Returns zero element
template<size_t n, const size_t & mod>
void Field<n,mod>::zero()
{
    Field<n,mod> res;
    res.intermediate_representation = 0;
    return res;
}

//Returns one element
template<size_t n, const size_t & mod>
void Field<n,mod>::one()
{
    Field<n,mod> res;
    res.intermediate_representation = 1;
    return res;
}

//Returns true iff this element is zero
template<size_t n, const size_t & mod>
bool Field<n,mod>::is_zero(const Field<n, mod> & fld)
{
    return fld.intermediate_representation == 0;
}

//Squares this element
template<size_t n, const size_t & mod>
void Field<n,mod>::square(Field<n, mod> & fld)
{
    //TODO implement
}

//Doubles this element
template<size_t n, const size_t & mod>
void Field<n,mod>::double(Field<n, mod> & fld)
{
    //TODO implement
}

//Negates this element
template<size_t n, const size_t & mod>
void Field<n,mod>::negate(Field<n, mod> & fld)
{
    //TODO implement
}

//Adds two elements
template<size_t n, const size_t & mod>
void Field<n,mod>::add(Field<n, mod> & fld1, const Field<n, mod> & fld2)
{
    //TODO implement
}

//Subtract element two from element one
template<size_t n, const size_t & mod>
void Field<n,mod>::substract(Field<n, mod> & fld1, const Field<n, mod> & fld2)
{
    //TODO implement
}

//Multiply two elements
template<size_t n, const size_t & mod>
void Field<n,mod>::mul(Field<n, mod> & fld1, const Field<n, mod> & fld2)
{
    //TODO implement
}

//Computes the multiplicative inverse of this element, if nonzero
template<size_t n, const size_t & mod>
void Field<n,mod>::mul_inv(Field<n, mod> & fld1)
{
    //TODO implement
}

//Exponentiates this element
template<size_t n, const size_t & mod>
void Field<n,mod>::pow(Field<n, mod> & fld1, const size_t pow)
{
    //TODO implement
}

}