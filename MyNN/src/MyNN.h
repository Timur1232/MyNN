#pragma once

#include <vector>
#include <span>
#include <string>
#include <functional>

namespace MyNN {

    using data_vector = std::vector<float>;
    using single_row = std::span<float>;
    using const_single_row = std::span<const float>;

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

        class const_RowIterator
        {
        public:
            const_RowIterator(const Matrix& mat, bool end = false)
                : m_Matrix(mat)
            {
                if (end)
                {
                    m_CurrentRow = mat.Rows;
                }
            }

            const_RowIterator operator++()
            {
                ++m_CurrentRow;
                return *this;
            }
            bool operator!=(const const_RowIterator& other)
            {
                return m_CurrentRow != other.m_CurrentRow;
            }
            const_single_row operator*()
            {
                return m_Matrix.GetRow(m_CurrentRow);
            }

        private:
            const Matrix& m_Matrix;
            size_t m_CurrentRow = 0;
        };

        float& At(size_t row, size_t col) { return Data[row * Rows + col]; }
        const float& At(size_t row, size_t col) const { return Data[row * Rows + col]; }

        single_row GetRow(size_t row) { return std::span(Data).subspan(row * Cols, Cols); }
        const_single_row GetRow(size_t row) const { return std::span(Data).subspan(row * Cols, Cols); }

        // Суммирует две матрицы; результат записывает в матрицу, которая вызывает метод
        void Sum(const Matrix& other);
        const Matrix& operator+=(const Matrix& other) { Sum(other); return *this; }

        RowIterator begin() { return RowIterator(*this); }
        RowIterator end() { return RowIterator(*this, true); }

        const_RowIterator begin() const { return const_RowIterator(*this); }
        const_RowIterator end() const { return const_RowIterator(*this, true); }
    };

    void fill(Matrix& mat, float val);
    void randomize(Matrix& mat, float min = 0.0f, float max = 1.0f);

    Matrix sum(const Matrix& lhs, const Matrix& rhs);
    void dot(Matrix& dst, const Matrix& lhs, const Matrix& rhs);
    //Matrix dot(const Matrix& lhs, const Matrix& rhs);

    void print(const Matrix& mat, std::string_view name);
    #define PRINT_MATRIX(mat) print(mat, #mat)

    /*=======================================================================*/

    class NNLayer
    {
    public:
        NNLayer(size_t neuronCount, size_t inputsCount)
            : m_ActivationField(1, neuronCount),
              m_Biases(1, neuronCount),
              m_WeightedConnections(inputsCount, neuronCount)
        {
        }

        void Randomize(float min = 0.0f, float max = 1.0f);

        const Matrix& GetOutputData() const { return m_ActivationField; }
        void Forward(const std::vector<float>& in, const std::function<float(float)>& activationFunc);

    private:
        Matrix m_ActivationField;
        Matrix m_Biases;
        Matrix m_WeightedConnections;
    };

    class NeuralNetwork
    {
    public:
        std::function<float(float)> ActivationFunc;
        float Epsilon = 1e-1f;
        float LearnRate = 1e-1f;

    public:
        NeuralNetwork(size_t inputFieldCount, const std::vector<size_t>& neuronsInLayers,
            std::function<float(float)>&& activationFunc);

        void PropagateForward(const std::vector<float>& in);
        float CalculateCost(const Matrix& trainData, const std::vector<float>& desired);
        void ForwardDifference(const Matrix& in, const std::vector<float>& desired);

    private:
        std::vector<NNLayer> m_Layers;
    };

} // MyNN