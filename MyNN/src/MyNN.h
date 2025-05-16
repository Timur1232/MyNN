#pragma once

#include <vector>
#include <span>
#include <string>
#include <functional>

namespace MyNN {

    using data_vector = std::vector<double>;
    using single_row = std::span<double>;

    struct Matrix
    {
        data_vector Data;
        size_t Rows;
        size_t Cols;
    
        Matrix(Matrix&&) = default;
        Matrix(const Matrix&) = default;

        Matrix(size_t rows, size_t cols)
            : Data(rows * cols), Rows(rows), Cols(cols)
        {
        }

        class RowIterator
        {
        public:
            RowIterator(Matrix& mat, bool end = false)
                : m_Matrix(mat)
            {
                if (end)
                {
                    m_CurrentRow = mat.Rows;
                }
            }

            RowIterator operator++()
            {
                ++m_CurrentRow;
                return *this;
            }
            bool operator!=(const RowIterator& other)
            {
                return m_CurrentRow != other.m_CurrentRow;
            }
            single_row operator*()
            {
                return m_Matrix.GetRow(m_CurrentRow);
            }

        private:
            Matrix& m_Matrix;
            size_t m_CurrentRow = 0;
        };

        double& At(size_t row, size_t col) { return Data[row * Cols + col]; }
        single_row GetRow(size_t row) { return std::span(Data).subspan(row * Cols, Cols); }

        // Суммирует две матрицы; результат записывает в матрицу, которая вызывает метод
        void Sum(Matrix& other);
        const Matrix& operator+=(Matrix& other) { Sum(other); return *this; }

        RowIterator begin() { return RowIterator(*this); }
        RowIterator end() { return RowIterator(*this, true); }
    };

    void fill(Matrix& mat, double val);
    void randomize(Matrix& mat, double min = 0.0f, double max = 1.0f);

    void dot(Matrix& dst, Matrix& lhs, Matrix& rhs);
    void dot_v_m(Matrix& dst, const single_row& lhs, Matrix& rhs);

    void print(Matrix& mat, std::string_view name);
    #define PRINT_MATRIX(mat) print(mat, #mat)

    /*=======================================================================*/

    struct NNLayer
    {
        Matrix ActivationField;
        Matrix Biases;
        Matrix WeightedConnections;

        NNLayer(size_t neuronCount, size_t inputsCount)
            : ActivationField(1, neuronCount),
              Biases(1, neuronCount),
              WeightedConnections(inputsCount, neuronCount)
        {
        }

        void Randomize(double min = 0.0f, double max = 1.0f);

        Matrix& GetOutputData() { return ActivationField; }
        void Forward(const single_row& in, const std::function<double(double)>& activationFunc);
    };

    class NeuralNetwork
    {
    public:
        std::function<double(double)> ActivationFunc;
        double Epsilon = 2e-1f;
        double LearnRate = 2e-1f;

    public:
        NeuralNetwork(const std::vector<size_t>& neuronsInLayers,
            std::function<double(double)>&& activationFunc);

        void PropagateForward(const single_row& in);
        double CalculateCost(Matrix& trainData, data_vector& desired);
        void ForwardDifference(Matrix& trainData, data_vector& desired, NeuralNetwork& grad);

        Matrix& GetOutput() { return m_Layers.back().GetOutputData(); }

        void Randomize(double min = 0.0f, double max = 1.0f);
        void PrintInfo() const;

    private:
        std::vector<NNLayer> m_Layers;
    };

} // MyNN