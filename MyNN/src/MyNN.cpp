#include "MyNN.h"

#include <algorithm>
#include <ranges>
#include <cassert>
#include <iostream>
#include <format>

namespace MyNN {

    void Matrix::Sum(const Matrix& other)
    {
        assert(other.Rows == Rows);
        assert(other.Cols == Cols);

        auto thisVal = Data.begin();
        auto otherVal = other.Data.begin();
        while (thisVal != Data.end() || otherVal != other.Data.end())
        {
            *thisVal += *otherVal;
            ++thisVal;
            ++otherVal;
        }
    }

    void fill(Matrix& mat, float val)
    {
        std::ranges::fill(mat.Data.begin(), mat.Data.end(), val);
    }

    void randomize(Matrix& mat, float min, float max)
    {
        std::ranges::for_each(mat.Data.begin(), mat.Data.end(),
            [min, max](float& val)
            {
                val = (float)std::rand() / (float)RAND_MAX * (max - min) + min;
            });
    }

    Matrix sum(const Matrix& lhs, const Matrix& rhs)
    {
        Matrix result(lhs);
        result.Sum(rhs);
        return result;
    }

    void dot(Matrix& dst, const Matrix& lhs, const Matrix& rhs)
    {
        assert(lhs.Cols == rhs.Rows);
        assert(dst.Rows == lhs.Rows);
        assert(dst.Cols == rhs.Cols);

        for (size_t i = 0; i < lhs.Rows; ++i)
        {
            for (size_t j = 0; j < rhs.Cols; ++j)
            {
                for (size_t k = 0; k < lhs.Cols; ++k)
                {
                    dst.At(i, j) += lhs.At(i, k) * rhs.At(k, j);
                }
            }
        }
    }

    /*Matrix dot(const Matrix& lhs, const Matrix& rhs)
    {
        assert(lhs.Cols == rhs.Rows);

        Matrix result(lhs.Rows, rhs.Cols);
        dot(result, lhs, rhs);
        return result;
    }*/

    void print(const Matrix& mat, std::string_view name)
    {
        std::cout << name << " = [\n    ";
        for (size_t row = 0; row < mat.Rows; ++row)
        {
            for (auto val : mat.GetRow(row))
            {
                std::cout << std::format("{:<6.3f} ", val);
            }
            if (row != mat.Rows - 1)
                std::cout << "\n    ";
            else
                std::cout << '\n';
        }
        std::cout << "]\n";
    }

    /*=======================================================================*/

    void NNLayer::Randomize(float min, float max)
    {
        randomize(m_Biases, min, max);
    }

    void NNLayer::Calculate(const Matrix& in, const std::function<float(float)>& activationFunc)
    {
        dot(m_ActivationField, in, m_WeightedConnections);
        m_ActivationField += m_Biases;
    }

} // MyNN