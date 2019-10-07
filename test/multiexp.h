#include <vector>

template <typename FieldT, typename FieldMul>
FieldT multi_exp(std::vector<FieldT> &a, std::vector<FieldMul> &b)
{
    FieldT result = FieldT::zero();
//#pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++)
    {
        FieldT::print(a[i]);
        FieldMul::print(b[i]);
        FieldT::print(a[i] * b[i]);
        FieldT::print(result);
        result = result + (a[i] * b[i]);
        FieldT::print(result);
    }
    return result;
}