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

#include "device_field_operators.h"

namespace fields{

    void testAdd()
    {
        fields::Field f1(1234);
        fields::Field f2(1234);
        fields::Field result(2468);
        add(f1,f2);
        assert(f1 == f2);
    }

    void testSubstract()
    {
        fields::Field f1(1234);
        fields::Field f2(1234);
        substract(f1,f2);
        assert(f1 == fields::Field::zero());
        fields::Field f3(1235);
        substract(f3, f2);
        assert(f3 == fields::Field::one());
    }

    void testModulo()
    {
        fields::Field f1(0);
        fields::Field f2(1234);
        
        fields::Field f3()
    }

    void testConstructor()
    {
        fields::Field f3(1);
        assert(f3 == fields::Field::one());
        fields::Field f4();
        assert(f4 == fields::Field::zero());
        fields::Field f5(0);
        assert(f5 == fields::Field::zero());

        fields::Field f1();
        fields::Field f2(1234);
        add(f1, fields::Field(1234));
        assert(f1 == f2);
        uint32_t[SIZE] tmp ={0,0,0,0,0,0,0,1234};
        fields::Field f6(tmp);
        assert(f6 == f2);
    }

    public static void main(int argc, char* argv)
    {

    }



}