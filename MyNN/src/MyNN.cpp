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

    void NNLayer::Forward(const std::vector<float>& in, const std::function<float(float)>& activationFunc)
    {
        dot(m_ActivationField, in, m_WeightedConnections);
        m_ActivationField += m_Biases;
        for (auto& x : m_ActivationField.Data)
        {
            x = activationFunc(x);
        }
    }

    NeuralNetwork::NeuralNetwork(size_t inputFieldCount, const std::vector<size_t>& neuronsInLayers,
        std::function<float(float)>&& activationFunc)
        : ActivationFunc(std::forward<std::function<float(float)>>(activationFunc))
    {
        // "входной слой"
        m_Layers.emplace_back(inputFieldCount, neuronsInLayers.front());

        for (size_t i = 1; i < neuronsInLayers.size(); ++i)
        {
            m_Layers.emplace_back(neuronsInLayers.at(i - 1), neuronsInLayers.at(i));
        }
    }

    void NeuralNetwork::PropagateForward(const std::vector<float>& in)
    {
        m_Layers.front().Forward(in, ActivationFunc);
        for (size_t i = 1; i < m_Layers.size(); ++i)
        {
            m_Layers[i].Forward(m_Layers.at(i - 1).GetOutputData(), ActivationFunc);
        }
    }

    float NeuralNetwork::CalculateCost(const Matrix& trainData, const std::vector<float>& desired)
    {
        const auto& nnOutput = m_Layers.back().GetOutputData().Data;
        assert(nnOutput.size() == desired.size());

        float cost = 0.0f;
        for (const auto& in : trainData)
        {
            PropagateForward(in);
            for (size_t i = 0; i < nnOutput.size(); ++i)
            {
                float dist = nnOutput.at(i) - desired.at(i);
                cost += dist * dist;
            }
        }
        return cost / trainData.Rows;
    }

    void NeuralNetwork::ForwardDifference(const Matrix& in, const std::vector<float>& desired)
    {

    }

} // MyNN