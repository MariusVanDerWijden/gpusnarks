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

#define DEBUG

#include "device_field.h"
#include "device_field_operators.h"
#include <assert.h>
#include <gmp.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <ctime>  

namespace fields{

    enum operand {add, substract, mul, pow};

    void testAdd()
    {
        printf("Addition test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        fields::Scalar result(2468);
        f1 =  f1 + f2;
        Scalar::testEquality(f1, result);
        printf("successful\n");
    }

    void test_subtract()
    {
        printf("_subtraction test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        f1 = f1 - f2;
        Scalar::testEquality(f1, fields::Scalar::zero());
        fields::Scalar f3(1235);
        f3 = f3 - f2;
        Scalar::testEquality(f3,fields::Scalar::one());
        printf("successful\n");
    }

    void testMultiply()
    {
        printf("Multiply test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(1522756));
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(1879080904));
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(3798462992)); 
        fields::Scalar f3(1234);
        f3 = f3 * f3;
        Scalar::testEquality(f3, fields::Scalar(1522756));
        printf("successful\n");
    }

    void testModulo()   
    {
        printf("Modulo test: ");
        fields::Scalar f1(uint32_t(0));
        fields::Scalar f2(1234);
        
        fields::Scalar f3();
        printf("successful\n");
    }

    void testPow()
    {
        printf("Scalar::pow test: ");
        fields::Scalar f1(2);
        Scalar::pow(f1, 0);
        Scalar::testEquality(f1, fields::Scalar::one());
        fields::Scalar f2(2);
        Scalar::pow(f2, 2);
        Scalar::testEquality(f2, fields::Scalar(4));
        Scalar::pow(f2, 10);
        Scalar::testEquality(f2, fields::Scalar(1048576));
        fields::Scalar f3(2);
        fields::Scalar f4(1048576);
        Scalar::pow(f3, 20);
        Scalar::testEquality(f3, f4);
        printf("successful\n");

    }

    void testConstructor()
    {
        printf("Constructor test: ");
        fields::Scalar f3(1);
        Scalar::testEquality(f3, fields::Scalar::one());
        fields::Scalar f4;
        Scalar::testEquality(f4, fields::Scalar::zero());
        fields::Scalar f5(uint32_t(0));
        Scalar::testEquality(f5, fields::Scalar::zero());

        fields::Scalar f1;
        fields::Scalar f2(1234);
        f1 = f1 +  fields::Scalar(1234);
        Scalar::testEquality(f1, f2);
        uint32_t tmp [SIZE];
        for(int i = 0; i < SIZE; i++)
            tmp[i] = 0;
        tmp[SIZE -1 ] = 1234;
        fields::Scalar f6(tmp);
        Scalar::testEquality(f6, f2);
        printf("successful\n");
    }

    

    void setMod()
    { /* 
        assert(SIZE == 24);
        _mod[0] = 115910;
        _mod[1] = 764593169;
        _mod[2] = 270700578; 
        _mod[3] = 4007841197; 
        _mod[4] = 3086587728;
        _mod[5] = 1536143341;
        _mod[6] = 1589111033;
        _mod[7] = 1821890675;
        _mod[8] = 134068517; 
        _mod[9] = 3902860685;
        _mod[10] = 2580620505;
        _mod[11] = 2707093405;
        _mod[12] = 2971133814;
        _mod[13] = 4061660573;
        _mod[14] = 3087994277;
        _mod[15] = 3411246648;
        _mod[16] = 1750781161;
        _mod[17] = 1987204260;
        _mod[18] = 1669861489;
        _mod[19] = 2596546032;
        _mod[20] = 3818738770;
        _mod[21] = 752685471;
        _mod[22] = 1586521054;
        _mod[23] = 610172929; */

        assert(SIZE == 24);
        for(int i = 0; i < SIZE; i ++)
        {
            _mod[i] = 0;
        }
        _mod[SIZE - 3] = 1;
    }

    void operate(fields::Scalar & f1, fields::Scalar const f2, int const op)
    {
        switch(op){
            case 0:
                f1 = f1 + f2; break;
            case 1:
                f1 = f1 - f2; break;
            case 2:
                f1 = f1 * f2; break;
            case 3:
                Scalar::pow(f1, (f2.im_rep[SIZE - 1] & 65535)); 
                break;
            default: break;
        } 
    }

    void operate(mpz_t mpz1, mpz_t const mpz2, mpz_t const mod, int const op)
    {
        switch(op){
            case 0:
                mpz_add(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod);
                break;
            case 1:
                mpz_sub(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod); 
                break;
            case 2:
                mpz_mul(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod);
                break;
            case 3:
                mpz_t pow;
                mpz_init(pow);
                mpz_set_ui(pow, 65535);
                mpz_and(pow, mpz2, pow);
                mpz_powm(mpz1, mpz1, pow, mod);
                mpz_clear(pow);
                break;
            default: break;
        }
    }

    void toMPZ(mpz_t ret, fields::Scalar f)
    {
        mpz_init(ret);
        mpz_import(ret, SIZE, 1, sizeof(uint32_t), 0, 0, f.im_rep);   
    }

    void compare(fields::Scalar f1, fields::Scalar f2, mpz_t mpz1, mpz_t mpz2, mpz_t mod, int op)
    {
        mpz_t tmp1;
        mpz_init_set(tmp1, mpz1);
        operate(f1, f2, op);
        operate(mpz1, mpz2, mod, op);
        mpz_t tmp;
        toMPZ(tmp, f1);
        if(mpz_cmp(tmp, mpz1) != 0){
            printf("Missmatch: ");
            gmp_printf ("t: %d [%Zd] %d [%Zd] \n",omp_get_thread_num(), tmp1, op, mpz2);
            gmp_printf ("t: %d CPU: [%Zd] GPU: [%Zd] \n",omp_get_thread_num() , mpz1, tmp);
            Scalar::printScalar(f1);
            assert(!"error");
        }
        mpz_clear(tmp1);
        mpz_clear(tmp);
    }

    void calculateModPrime()
    {
        mpz_t one, minus1, mod_prime, mod, base;
        mpz_init(mod);
        mpz_init(minus1);
        mpz_init(mod_prime);
        mpz_init(base);
        mpz_init(one);
        mpz_set_si(minus1, -1);
        mpz_set_si(one, 1);
        mpz_set_si(base, 4294967296);
        mpz_set_str(mod, "18446744073709551616", 0);
        mpz_mul(mod_prime, minus1, mod);
        mpz_div(mod_prime, one, mod_prime);
        mpz_mod(mod_prime, mod_prime, base);
        gmp_printf ("Mod_prime:  [%Zd] \n",mod_prime);        
    }

    void fuzzTest()
    {
        printf("Fuzzing test: ");
        
        size_t i_step = 12345671;
        size_t k_step = 76543210;
        auto start = std::chrono::system_clock::now();
    
        //#pragma omp parallel for
        for(size_t i = 0; i < 4294967295; i = i + i_step)
        {
            if(omp_get_thread_num() == 0){
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;

                printf("%f%% %d sec \n", (float(i) / 4294967295) * omp_get_num_threads(), (int)elapsed_seconds.count());
            }
            mpz_t a, b, mod;
            mpz_init(a);
            mpz_init(b);
            mpz_init(mod);
            mpz_set_str(mod, "18446744073709551616", 0);
            //mpz_set_str(mod, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
            mpz_set_ui(b, i);
            fields::Scalar f2(i);
            for(size_t k = 0; k < 4294967295; k = k + k_step)
            {
                for(size_t z = 0; z <= 3; z++ )
                {
                    mpz_set_ui(a, k);
                    fields::Scalar f1(k);
                    compare(f1,f2,a,b,mod,z);
                }
            }
            mpz_clear(a);
            mpz_clear(b);
            mpz_clear(mod);
        }
        printf("successful\n");
    }
}

int main(int argc, char** argv)
{
    fields::calculateModPrime();
    
    fields::setMod();
    fields::testConstructor();
    fields::testAdd();
    fields::test_subtract();
    fields::testMultiply();
    fields::testPow();
    fields::fuzzTest();
    printf("\nAll tests successful\n");
    return 0;
}



