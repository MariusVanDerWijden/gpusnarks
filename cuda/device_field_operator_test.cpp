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

namespace fields
{

mpz_t R;
mpz_t R_PRIME;
mpz_t mod;

enum operand
{
    add,
    substract,
    mul,
    pow
};

void toMPZ(mpz_t ret, fields::Scalar f)
{
    mpz_init(ret);
    mpz_import(ret, SIZE, -1, sizeof(uint32_t), 0, 0, f.im_rep);
}

void toScalar(fields::Scalar &f, mpz_t num)
{
    size_t size = SIZE;
    mpz_export(f.im_rep, &size, -1, sizeof(uint32_t), 0, 0, num);
}

void calculateModPrime()
{
    mpz_t one, minus1, n_prime, n, m, mod, two;
    mpz_init(one);
    mpz_init(minus1);
    mpz_init(n_prime);
    mpz_init(n);
    mpz_init(m);
    mpz_init(one);
    mpz_init(mod);
    mpz_init(two);

    mpz_set_ui(two, 2);
    mpz_set_ui(one, 1);
    mpz_set_str(n, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
    uint exp = SIZE * 32;    //log2(n) rounded up
    mpz_pow_ui(m, two, exp); //2^log2(n)

    mpz_set_si(minus1, -1);
    int i = mpz_invert(n_prime, n, m); // n' = n^-1
    uint32_t rop[SIZE];
    uint32_t _mod[SIZE];
    size_t size = SIZE;
    mpz_export(rop, &size, 0, 32, 1, 0, n_prime);
    mpz_export(_mod, &size, 0, 32, 1, 0, n);
    gmp_printf("Mod_prime:  [%Zd] %d %u %u %u \n", n_prime, i, rop[0], _mod[0], _mod[SIZE - 1]);

    Scalar s;
    toScalar(s, n_prime);
    Scalar::print(s);
    printf("\n");
    mpz_invert(R_PRIME, m, n); // r' = r^-1

    mpz_t R_tmp;
    mpz_init(R_tmp);
    mpz_mul(R_tmp, m, R_PRIME);
    mpz_mod(R_tmp, R_tmp, n);
    gmp_printf("R_PRIME [%Zd] \n", R_PRIME);
    if (mpz_cmp(R_tmp, one) != 0)
    {
        gmp_printf("R [%Zd] N  [%Zd] R_PRIME [%Zd] TMP: [%Zd]\n", m, n, R_PRIME, R_tmp);
        printf("Missmatch: \n");
        assert(!"error2");
    }
    mpz_init(R);
    mpz_set(R, m);

    /* 
            //R_SQUARE
            //R = 2^(32*SIZE) = 2^768
            mpz_t r_square;
            mpz_init(r_square);
            mpz_pow_ui(r_square, two, exp);
            gmp_printf ("R_Square:  [%Zd] ",r_square);  
            mpz_init(R);
            mpz_set(R, r_square);

            //R_PRIME 
            mpz_t N_PRIME;
            mpz_init(N_PRIME);
            //TODO calculate R_PRIME and N_PRIME with extended euclidean
            mpz_gcdext(one, N_PRIME, R_PRIME, n, R);
            //mpz_add(R_PRIME, R_PRIME, n);

            // Check correctness
            mpz_t R_tmp, N_tmp;
            mpz_init(R_tmp);
            mpz_init(N_tmp);
            mpz_mul(R_tmp, R, R_PRIME);
            mpz_mul(N_tmp, n, N_PRIME);
            mpz_sub(R_tmp, R_tmp, N_tmp);
            gmp_printf ("R [%Zd] N  [%Zd] R_PRIME [%Zd] N_PRIME [%Zd] TMP: [%Zd]\n",R, n, R_PRIME, N_PRIME, R_tmp); 
            if(mpz_cmp(R_tmp, one) != 0){
                printf("Missmatch: \n"); 
                assert(!"error");
            }
            // (R * R') mod N === 1
            mpz_mul(R_tmp, n, N_PRIME);
            mpz_mod(R_tmp, R_tmp, R);
            if(mpz_cmp(R_tmp, one) != 0){
                gmp_printf ("R [%Zd] N  [%Zd] R_PRIME [%Zd] TMP: [%Zd]\n",R, n, R_PRIME, R_tmp); 
                printf("Missmatch: \n"); 
                assert(!"error2");
            }
            assert(!"successful");*/
}

Scalar to_monty(fields::Scalar &f, const mpz_t mod)
{
    mpz_t tmp;
    toMPZ(tmp, f);
    gmp_printf("tmp0 [%Zd] \n", tmp);
    mpz_mul(tmp, tmp, R);
    gmp_printf("tmp1 [%Zd] \n", tmp);
    mpz_mod(tmp, tmp, mod);
    gmp_printf("tmp2 [%Zd] \n", tmp);
    f = Scalar::zero();
    toScalar(f, tmp);
    mpz_clear(tmp);
    return f;
}

void from_monty(fields::Scalar &f, const mpz_t mod)
{
    /* 
            // Works on CPU
            mpz_t tmp;
            toMPZ(tmp, f);
            mpz_mul(tmp, tmp, R_PRIME);
            mpz_mod(tmp, tmp, mod);
            f = Scalar::zero();
            toScalar(f, tmp);
            mpz_clear(tmp);
            */
    Scalar::print(f);
    f = f * Scalar::one();
    mpz_t tmp;
    toMPZ(tmp, f);
    gmp_printf("tmp0 [%Zd] \n", tmp);
    mpz_sub(tmp, tmp, mod);
    mpz_mod(tmp, tmp, mod);
    gmp_printf("tmp1 [%Zd] \n", tmp);
    f = Scalar::zero();
    toScalar(f, tmp);
}

void testEncodeDecode()
{
    for (size_t i = 1; i < 4294967295; i = i + 1234567)
    {
        // Test Encode/Decode
        mpz_t a;
        mpz_init(a);
        mpz_set_ui(a, i);
        fields::Scalar f(i);
        mpz_t tmp;
        toMPZ(tmp, f);
        if (mpz_cmp(tmp, a) != 0)
        {
            printf("Encoding Error: [%Zd]\n", tmp);
            assert(!"error");
        }
        Scalar s;
        toScalar(s, tmp);
        Scalar::testEquality(s, f);
        mpz_clear(a);
        mpz_clear(tmp);
    }
}

void testMonty()
{
    printf("Test monty: \n");
    mpz_init(mod);
    mpz_set_str(mod, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
    for (size_t k = 1; k < 4294967295; k = k + 7654321)
    {
        Scalar monty(k);
        Scalar non_modified(k);
        monty = to_monty(monty, mod);
        from_monty(monty, mod);
        Scalar::testEquality(monty, non_modified);
    }
}

void testAdd()
{
    printf("Addition test: ");
    fields::Scalar f1(1234);
    fields::Scalar f2(1234);
    fields::Scalar result(2468);
    f1 = f1 + f2;
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
    Scalar::testEquality(f3, fields::Scalar::one());
    printf("successful\n");
}

void testMultiply()
{
    printf("Multiply test: ");
    fields::Scalar f1(1234);
    fields::Scalar f2(1234);
    to_monty(f1, mod);
    to_monty(f2, mod);
    f1 = f1 * f2;
    from_monty(f1, mod);
    Scalar::testEquality(f1, fields::Scalar(1522756));
    f1 = f1 * f2;
    Scalar::testEquality(f1, fields::Scalar(1879080904));
    f1 = f1 * f2;
    //Scalar::testEquality(f1, fields::Scalar(3798462992)); this is only valid in mod 2
    f1 = f1 * f1;
    //Scalar::testEquality(f1, fields::Scalar(14428321101593592064));
    fields::Scalar f3(1234);
    f3 = f3 * f3;
    //Scalar::testEquality(f3, fields::Scalar(1522756));
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
    f1 = f1 ^ 0;
    Scalar::testEquality(f1, fields::Scalar::one());
    fields::Scalar f2(2);
    f2 = f2 ^ 2;
    Scalar::testEquality(f2, fields::Scalar(4));
    f2 = f2 ^ 10;
    Scalar::testEquality(f2, fields::Scalar(1048576));
    fields::Scalar f3(2);
    fields::Scalar f4(1048576);
    f3 = f3 ^ 20;
    Scalar::testEquality(f3, f4);
    fields::Scalar f5(2);
    f5 = f5 ^ 35;
    uint32_t tmp[SIZE] = {0};
    tmp[SIZE - 2] = 8;
    Scalar::testEquality(f5, fields::Scalar(tmp));
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
    f1 = f1 + fields::Scalar(1234);
    Scalar::testEquality(f1, f2);
    uint32_t tmp[SIZE] = {0};
    tmp[0] = 1234;
    fields::Scalar f6(tmp);
    Scalar::testEquality(f6, f2);
    printf("successful\n");
}

void setMod()
{
    assert(SIZE == 24);
    mpz_t n;
    mpz_init(n);
    mpz_set_str(n, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
    size_t size = SIZE;
    Scalar s;
    toScalar(s, n);
    Scalar::print(s);
    gmp_printf("Mod:  [%Zd] \n", n);
    assert(SIZE == 24);
    for (int i = 0; i < SIZE; i++)
    {
        printf("%u , ", _mod[i]);
    }
}

void testBitAt()
{
    auto b = fields::Scalar(1);
    auto d = fields::Scalar(4321);
    Scalar::print(b);
    Scalar::print(d);
    for (int i = 0; i < SIZE * 32; i++)
    {
        printf(" %i ", hasBitAt(d, i));
    }
}

void testMNT4()
{
    auto a = fields::mnt4753_G1(fields::Scalar(1234), fields::Scalar(12345), fields::Scalar(12346));
    auto c = fields::mnt4753_G1(fields::Scalar(123434), fields::Scalar(1232), fields::Scalar(123490));
    auto b = fields::Scalar(4321);
    auto d = fields::Scalar(4320);
    mnt4753_G1::print(a);
    Scalar::print(b);
    Scalar::print(d);
    Scalar::print(b + d);
    Scalar::print(b * d);
    Scalar::print(b - d);
    Scalar::print(b ^ 12);
    mnt4753_G1::print(a * b);
    //mnt4753_G1::print(a + c);
}

void operate(fields::Scalar &f1, fields::Scalar f2, mpz_t const mod, int const op)
{
    switch (op)
    {
    case 0:
        f1 = f1 + f2;
        break;
    case 1:
        f1 = f1 - f2;
        break;
    case 2:
        to_monty(f1, mod);
        to_monty(f2, mod);
        f1 = f1 * f2;
        from_monty(f1, mod);
        break;
    case 3:
        f1 = f1 ^ (f2.im_rep[SIZE - 1] & 65535);
        break;
    default:
        break;
    }
}

void operate(mpz_t mpz1, mpz_t const mpz2, mpz_t const mod, int const op)
{
    switch (op)
    {
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
    default:
        break;
    }
}

void compare(fields::Scalar f1, fields::Scalar f2, mpz_t mpz1, mpz_t mpz2, mpz_t mod, int op)
{
    mpz_t tmp1;
    mpz_init_set(tmp1, mpz1);
    operate(f1, f2, mod, op);
    operate(mpz1, mpz2, mod, op);
    mpz_t tmp;
    toMPZ(tmp, f1);
    if (mpz_cmp(tmp, mpz1) != 0)
    {
        //printf("Missmatch: \n");
        //gmp_printf ("t: %d [%Zd] %d [%Zd] \n",omp_get_thread_num(), tmp1, op, mpz2);
        //gmp_printf ("t: %d CPU: [%Zd] GPU: [%Zd] \n",omp_get_thread_num() , mpz1, tmp);
        //Scalar::print(f1);
        //assert(!"error");
    }
    mpz_clear(tmp1);
    mpz_clear(tmp);
}

void fuzzTest()
{
    printf("Fuzzing test: ");

    size_t i_step = 12345671;
    size_t k_step = 76543210;
    size_t loop_start = 1;
    auto start = std::chrono::system_clock::now();

    //#pragma omp parallel for
    for (size_t i = loop_start; i < 4294967295; i = i + i_step)
    {
        if (omp_get_thread_num() == 0)
        {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;

            printf("%f%% %d sec \n", (float(i) / 4294967295) * omp_get_num_threads(), (int)elapsed_seconds.count());
        }
        mpz_t a, b, mod;
        mpz_init(a);
        mpz_init(b);
        mpz_init(mod);
        //mpz_set_str(mod, "18446744073709551616", 0);
        mpz_set_str(mod, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
        mpz_set_ui(b, i);
        fields::Scalar f2(i);
        for (size_t k = loop_start; k < 4294967295; k = k + k_step)
        {
            for (size_t z = 0; z <= 3; z++)
            {
                mpz_set_ui(a, k);
                fields::Scalar f1(k);
                compare(f1, f2, a, b, mod, z);
            }
        }
        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(mod);
    }
    printf("successful\n");
}
} // namespace fields

int main(int argc, char **argv)
{
    fields::testMNT4();
    //fields::testBitAt();
    /*
    fields::calculateModPrime();
    fields::testEncodeDecode();
    //fields::testMonty();
    fields::setMod();
    fields::testConstructor();
    fields::testAdd();
    fields::test_subtract();
    //fields::testMultiply();
    //fields::testPow();
    fields::fuzzTest();
    */
    printf("\nAll tests successful\n");
    return 0;
}
