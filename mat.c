#include <string.h>
#include "mat.h"

mat_elem_t rand_float(void)
{
    return (mat_elem_t)rand() / (mat_elem_t)RAND_MAX;
}

mat_elem_t sigmoidf(mat_elem_t x)
{
    return 1.f / (1.f + expf(-x));
}

mat_elem_t dsigmoidf(mat_elem_t x)
{
    mat_elem_t s = sigmoidf(x);
    return s * (1.f - s);
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = malloc(sizeof(mat_elem_t) * rows * cols);
    assert(m.es != NULL);
    return m;
}

Mat mat_resize(Mat m, size_t rows, size_t cols)
{
    if (m.rows == rows && m.cols == cols)
        return m;
    
    mat_free(m);
    return mat_alloc(rows, cols);
}

Mat mat_free(Mat m)
{
    if (!m.es)
        return (Mat){0};
    assert(m.es != NULL);
    free(m.es);
    return (Mat){0};
}

void mat_zero(Mat m)
{
    assert(m.es);
    memset(m.es, 0, sizeof(*m.es) * m.rows * m.cols);
}

void mat_fill(Mat m, mat_elem_t x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, mat_elem_t low, mat_elem_t high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

Mat mat_row_view(Mat m, size_t row)
{
    return mat_rows_view(m, row, 1);
}

Mat mat_rows_view(Mat m, size_t i, size_t rows)
{
    return (Mat){.rows = rows, .cols = m.cols, .es = &MAT_AT(m, i, 0)};
}

void mat_copy_inline(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_dot_inline(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0.f;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_dot(Mat a, Mat b)
{
    Mat m = mat_alloc(a.rows, b.cols);
    mat_dot_inline(m, a, b);
    return m;
}

Mat mat_transpose(Mat a)
{
    Mat m = mat_alloc(a.cols, a.rows);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(m, j, i) = MAT_AT(a, i, j);
        }
    }
    return m;
}

// C = A^T * B
// A: (rowsA x colsA)
// B: (rowsB x colsB)
// C: (colsA x colsB)
void mat_dot_transposeA_inline(Mat C, Mat A, Mat B)
{
    assert(A.rows == B.rows);
    assert(C.rows == A.cols);
    assert(C.cols == B.cols);

    size_t m = A.cols; // A^Tの行数
    size_t n = B.cols; // Bの列数
    size_t k = A.rows; // A^Tの列数 = Bの行数

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            float sum = 0.f;
            for (size_t t = 0; t < k; t++)
            {
                sum += MAT_AT(A, t, i) * MAT_AT(B, t, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void mat_sum_inline(Mat dst, Mat a)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sum_bias_inline(Mat dst, Mat bias)
{
    assert(bias.rows == 1);
    assert(dst.cols == bias.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(bias, 0, j);
        }
    }
}

void mat_scalar_div_inline(Mat dst, float scalar)
{
    assert(scalar != 0.f); // 0で割らないように注意
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) /= scalar;
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_dsig(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = dsigmoidf(MAT_AT(src, i, j));
        }
    }
}

void mat_softmax(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        float max = MAT_AT(m, 0, 0);
        for (size_t j = 1; j < m.cols; j++)
        {
            if (MAT_AT(m, i, j) > max)
                max = MAT_AT(m, i, j);
        }
        float sum = 0.f;
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - max);
            sum += MAT_AT(m, i, j);
        }
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) /= sum;
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}
