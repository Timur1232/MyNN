#pragma once

namespace MyNN {

    /*using data_vector = std::vector<float>;
    using single_row = std::span<float>;
    using const_single_row = std::span<const float>;*/

    class Matrix
    {
    public:
        Matrix(Matrix&&) = default;
        Matrix(const Matrix&) = default;
        Matrix(size_t rows, size_t cols)
            : m_Data(new double[rows * cols]), m_Rows(rows), m_Cols(cols)
        {
        }
        ~Matrix() { delete[] m_Data; }

        double& At(size_t row, size_t col) { return m_Data[row * m_Rows + col]; }
        const double& At(size_t row, size_t col) const { return m_Data[row * m_Rows + col]; }

        // Суммирует две матрицы; результат записывает в матрицу, которая вызывает метод
        void Sum(const Matrix& other);
        const Matrix& operator+=(const Matrix& other) { Sum(other); return *this; }

        /*single_row GetRow(size_t row) { return std::span(m_Data).subspan(row * m_Cols, m_Cols); }
        const_single_row GetRow(size_t row) const { return std::span(m_Data).subspan(row * m_Cols, m_Cols); }*/


        /*RowIterator begin() { return RowIterator(*this); }
        RowIterator end() { return RowIterator(*this, true); }

        const_RowIterator begin() const { return const_RowIterator(*this); }
        const_RowIterator end() const { return const_RowIterator(*this, true); }*/
    private:
        double* m_Data;
        size_t m_Rows;
        size_t m_Cols;
    };

    class MatrixView
    {
    public:
        MatrixView(Matrix& mat, );

    private:
        double* m_Data;
        size_t m_Rows;
        size_t m_Cols;
        size_t m_Stride;
    };

    void fill(Matrix& mat, float val);
    void randomize(Matrix& mat, float min = 0.0f, float max = 1.0f);

    Matrix sum(const Matrix& lhs, const Matrix& rhs);
    void dot(Matrix& dst, const Matrix& lhs, const Matrix& rhs);
    //Matrix dot(const Matrix& lhs, const Matrix& rhs);

    void print(const Matrix& mat, std::string_view name);
#define PRINT_MATRIX(mat) print(mat, #mat)

    class RowIterator
    {
    public:
        RowIterator(Matrix& mat, bool end = false)
            : m_Matrix(mat)
        {
            if (end)
            {
                m_CurrentRow = mat.m_Rows;
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
                m_CurrentRow = mat.m_Rows;
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

} // MyNN