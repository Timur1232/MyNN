#include "MyNN.h"

#include <algorithm>
#include <ranges>
#include <cassert>
#include <iostream>
#include <format>

namespace MyNN {

    void Matrix::Sum(Matrix& other)
    {
        assert(other.Rows == Rows);
        assert(other.Cols == Cols);

        for (size_t i = 0; i < Data.size(); ++i)
        {
            Data[i] = other.Data[i];
        }
    }

    void fill(Matrix& mat, double val)
    {
        std::ranges::fill(mat.Data.begin(), mat.Data.end(), val);
    }

    void randomize(Matrix& mat, double min, double max)
    {
        std::ranges::for_each(mat.Data.begin(), mat.Data.end(),
            [min, max](double& val)
            {
                val = (double)std::rand() / (double)RAND_MAX * (max - min) + min;
            });
    }

    void dot(Matrix& dst, Matrix& lhs, Matrix& rhs)
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

    void dot_v_m(Matrix& dst, const single_row& lhs, Matrix& rhs)
    {
        assert(lhs.size() == rhs.Rows);
        assert(dst.Rows == 1);
        assert(dst.Cols == rhs.Cols);

        for (size_t j = 0; j < rhs.Cols; ++j)
        {
            for (size_t k = 0; k < lhs.size(); ++k)
            {
                dst.At(0, j) += lhs[k] * rhs.At(k, j);
            }
        }
    }

    void print(Matrix& mat, std::string_view name)
    {
        std::cout << name << " = [\n    ";
        size_t rowIndex = 0;
        for (const auto& row : mat)
        {
            for (auto val : row)
            {
                std::cout << std::format("{:<6.3f} ", val);
            }
            if (rowIndex != mat.Rows - 1)
                std::cout << "\n    ";
            else
                std::cout << '\n';
            ++rowIndex;
        }
        std::cout << "]\n";
    }

    /*=======================================================================*/

    void NNLayer::Randomize(double min, double max)
    {
        randomize(Biases, min, max);
        randomize(WeightedConnections, min, max);
    }

    void NNLayer::Forward(const single_row& in, const std::function<double(double)>& activationFunc)
    {
        dot_v_m(ActivationField, in, WeightedConnections);
        ActivationField += Biases;
        for (auto& x : ActivationField.Data)
        {
            x = activationFunc(x);
        }
    }

    NeuralNetwork::NeuralNetwork(const std::vector<size_t>& neuronsInLayers,
        std::function<double(double)>&& activationFunc)
        : ActivationFunc(std::forward<std::function<double(double)>>(activationFunc))
    {
        assert(neuronsInLayers.size() >= 2);
        // "входной слой"
        m_Layers.emplace_back(neuronsInLayers.at(0), neuronsInLayers.at(0));

        for (size_t i = 1; i < neuronsInLayers.size(); ++i)
        {
            m_Layers.emplace_back(neuronsInLayers.at(i), neuronsInLayers.at(i - 1));
        }
    }

    void NeuralNetwork::PropagateForward(const single_row& in)
    {
        m_Layers.front().Forward(in, ActivationFunc);
        for (size_t i = 1; i < m_Layers.size(); ++i)
        {
            m_Layers[i].Forward(m_Layers.at(i - 1).GetOutputData().GetRow(0), ActivationFunc);
        }
    }

    double NeuralNetwork::CalculateCost(Matrix& trainData, data_vector& desired)
    {
        assert(trainData.Rows == desired.size());

        const auto& nnOutput = m_Layers.back().GetOutputData().Data;
        double cost = 0.0f;
        size_t row = 0;
        for (const auto& in : trainData)
        {
            PropagateForward(in);
            double dist = nnOutput.at(0) - desired.at(row);
            cost += dist * dist;
            ++row;
        }
        return cost / trainData.Rows;
    }

    void NeuralNetwork::ForwardDifference(Matrix& trainData, data_vector& desired, NeuralNetwork& grad)
    {
        assert(trainData.Rows == desired.size());

        double cost = CalculateCost(trainData, desired);

        for (size_t layer = 0; layer < m_Layers.size(); ++layer)
        {
            for (size_t i = 0; i < m_Layers[layer].WeightedConnections.Rows; ++i)
            {
                for (size_t j = 0; j < m_Layers[layer].WeightedConnections.Cols; ++j)
                {
                    double& weight = m_Layers[layer].WeightedConnections.At(i, j);
                    double savedWeight = weight;
                    weight += Epsilon;
                    grad.m_Layers[layer].WeightedConnections.At(i, j) = (CalculateCost(trainData, desired) - cost) / Epsilon;
                    weight = savedWeight;
                }
            }
            for (size_t j = 0; j < m_Layers[layer].Biases.Cols; ++j)
            {
                double& bias = m_Layers[layer].Biases.At(0, j);
                double savedBias = bias;
                bias += Epsilon;
                grad.m_Layers[layer].Biases.At(0, j) = (CalculateCost(trainData, desired) - cost) / Epsilon;
                bias = savedBias;
            }
        }

        for (size_t layer = 0; layer < m_Layers.size(); ++layer)
        {
            for (size_t i = 0; i < m_Layers[layer].WeightedConnections.Rows; ++i)
            {
                for (size_t j = 0; j < m_Layers[layer].WeightedConnections.Cols; ++j)
                {
                    m_Layers[layer].WeightedConnections.At(i, j) -= grad.m_Layers[layer].WeightedConnections.At(i, j) * LearnRate;
                }
            }
            for (size_t j = 0; j < m_Layers[layer].Biases.Cols; ++j)
            {
                m_Layers[layer].Biases.At(0, j) -= grad.m_Layers[layer].Biases.At(0, j) * LearnRate;
            }
        }
    }

    void NeuralNetwork::Randomize(double min, double max)
    {
        for (auto& layer : m_Layers)
            layer.Randomize(min, max);
    }

    void NeuralNetwork::PrintInfo() const
    {
        std::cout << "Layers count: " << m_Layers.size() << '\n';
        size_t i = 0;
        for (const auto& layer : m_Layers)
        {
            std::cout << std::format("layer #{}:\n  weights : {}x{}\n  biases : {}x{}\n  int: {}x{}\n",
                i, layer.WeightedConnections.Rows, layer.WeightedConnections.Cols,
                layer.Biases.Rows, layer.Biases.Cols,
                layer.ActivationField.Rows, layer.ActivationField.Cols);
            ++i;
        }
        std::cout << std::endl;
    }

} // MyNN