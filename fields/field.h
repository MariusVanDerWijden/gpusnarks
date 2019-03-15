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

//TODO second parameter should be big integer
template<size_t n, const size_t & mod>
class Field {

	public:
		//TODO should be big datatype
		size_t intermediate_representation;
		//Returns zero element
    	static Field<n, modulus> zero();
    	//Returns one element
    	static Field<n, modulus> one();
    	//Returns true iff this element is zero
    	static bool is_zero(const Field<n, mod> & fld);
    	//Squares this element
    	static void square(Field<n, mod> & fld);
    	//Doubles this element
    	static void double(Field<n, mod> & fld);
   	 	//Negates this element
    	static void negate(Field<n, mod> & fld);
    	//Adds two elements
    	static void add(Field<n, mod> & fld1, const Field<n, mod> & fld2);
    	//Subtract element two from element one
    	static void substract(Field<n, mod> & fld1, const Field<n, mod> & fld2);
    	//Multiply two elements
    	static void mul(Field<n, mod> & fld1, const Field<n, mod> & fld2);
    	//Computes the multiplicative inverse of this element, if nonzero.
    	static void mul_inv(Field<n, mod> & fld1);
    	//Exponentiates this element 
    	static void pow(Field<n, mod> & fld1, const size_t pow);
};
}