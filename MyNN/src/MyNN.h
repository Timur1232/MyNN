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

        float& At(size_t row, size_t col) { return Data[row * Rows + col]; }
        const float& At(size_t row, size_t col) const { return Data[row * Rows + col]; }

        single_row GetRow(size_t row) { return std::span(Data).subspan(row * Cols, Cols); }
        const_single_row GetRow(size_t row) const { return std::span(Data).subspan(row * Cols, Cols); }

        // Суммирует две матрицы; результат записывает в матрицу, которая вызывает метод
        void Sum(const Matrix& other);
        const Matrix& operator+=(const Matrix& other) { Sum(other); return *this; }
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
        void Calculate(const Matrix& in, const std::function<float(float)>& activationFunc);

    private:
        Matrix m_ActivationField;
        Matrix m_Biases;
        Matrix m_WeightedConnections;
    };

} // MyNN