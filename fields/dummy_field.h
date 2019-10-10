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
#include <cstdint>

namespace dummy_fields{

using size_t = decltype(sizeof 1ll);

class Field {

	public:
		//Intermediate representation
		uint32_t im_rep;
        //Modulo
        static uint32_t mod;
        //N
        static uint32_t n;
		 //Default constructor
    	Field() = default;
		//Construct from value
		Field(uint32_t value)
		{
			im_rep = value;
		} 
		//Returns zero element
    	static Field zero();
    	//Returns one element
    	static Field one();
    	//Returns true iff this element is zero
    	static bool is_zero(const Field & fld);
    	//Squares this element
    	static void square(Field & fld);
    	//Doubles this element
    	//static void double(Field & fld);
   	 	//Negates this element
    	static void negate(Field & fld);
    	//Adds two elements
    	static void add(Field & fld1, const Field & fld2);
    	//Subtract element two from element one
    	static void subtract(Field & fld1, const Field & fld2);
    	//Multiply two elements
    	static void mul(Field & fld1, const Field & fld2);
    	//Computes the multiplicative inverse of this element, if nonzero.
    	static void mul_inv(Field & fld1);
    	//Exponentiates this element 
    	static void pow(Field & fld1, const size_t pow);
};

}