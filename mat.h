#ifndef MAT_H_
#define MAT_H_
#include <sys/types.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef float mat_elem_t;

typedef struct
{
    size_t rows;
    size_t cols;
    mat_elem_t *data;
} Mat;

#define MAT_AT(m, row, col) ((m).data[(row) * (m).cols + (col)])
#define MAT_PRINT(m) mat_print(m, #m, 0)

Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat *m);
void mat_zero(Mat m);
void mat_rand(Mat m, mat_elem_t low, mat_elem_t high);
void mat_add_inplace(Mat dst, Mat other);
void mat_add_rowwise_inplace(Mat dst, Mat rowvec);
void mat_dot(Mat dst, Mat a, Mat b);
Mat mat_dot_alloc(Mat a, Mat b);
void mat_transpose(Mat dst, Mat src);
Mat mat_transpose_alloc(Mat dst);
void mat_sum_cols(Mat dst, Mat src);
Mat mat_sum_cols_alloc(Mat src);
void mat_print(Mat m, const char *name, size_t padding);

#ifdef MAT_IMPLEMENTATION
mat_elem_t rand_mat_elem()
{
    return (mat_elem_t)rand() / (mat_elem_t)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    mat_elem_t *data = malloc(sizeof(mat_elem_t) * rows * cols);
    if (!data)
    {
        perror("malloc failed");
        return (Mat){0};
    }
    return (Mat){.rows = rows, .cols = cols, .data = data};
}

void mat_free(Mat *m)
{
    if (m->data)
    {
        free(m->data);
    }
    *m = (Mat){0};
}

void mat_zero(Mat m)
{
    for (size_t row = 0; row < m.rows; row++)
    {
        for (size_t col = 0; col < m.cols; col++)
        {
            MAT_AT(m, row, col) = 0;
        }
    }
}

void mat_rand(Mat m, mat_elem_t low, mat_elem_t high)
{
    for (size_t row = 0; row < m.rows; row++)
    {
        for (size_t col = 0; col < m.cols; col++)
        {
            MAT_AT(m, row, col) = rand_mat_elem() * (high - low) + low;
        }
    }
}

void mat_add_inplace(Mat dst, Mat other)
{
    assert(dst.rows == other.rows);
    assert(dst.cols == other.cols);
    for (size_t row = 0; row < dst.rows; row++)
    {
        for (size_t col = 0; col < dst.cols; col++)
        {
            MAT_AT(dst, row, col) += MAT_AT(other, row, col);
        }
    }
}

void mat_add_rowwise_inplace(Mat dst, Mat rowvec)
{
    assert(dst.cols == rowvec.cols);
    for (size_t row = 0; row < dst.rows; row++)
    {
        for (size_t col = 0; col < dst.cols; col++)
        {
            MAT_AT(dst, row, col) += MAT_AT(rowvec, 0, col);
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    for (size_t row = 0; row < dst.rows; row++)
    {
        for (size_t col = 0; col < dst.cols; col++)
        {
            mat_elem_t sum = 0.0f;
            for (size_t inner = 0; inner < a.cols; inner++)
            {
                sum += MAT_AT(a, row, inner) * MAT_AT(b, inner, col);
            }
            MAT_AT(dst, row, col) = sum;
        }
    }
}

Mat mat_dot_alloc(Mat a, Mat b)
{
    Mat dst = mat_alloc(a.rows, b.cols);
    mat_dot(dst, a, b);
    return dst;
}

void mat_transpose(Mat dst, Mat src)
{
    assert(dst.cols == src.rows);
    assert(dst.rows == src.cols);
    for (size_t row = 0; row < src.rows; row++)
    {
        for (size_t col = 0; col < src.cols; col++)
        {
            assert(col < dst.rows);
            assert(row < dst.cols);
            MAT_AT(dst, col, row) = MAT_AT(src, row, col);
        }
    }
}

Mat mat_transpose_alloc(Mat src)
{
    Mat dst = mat_alloc(src.cols, src.rows);
    mat_transpose(dst, src);
    return dst;
}

void mat_sum_cols(Mat dst, Mat src)
{
    assert(dst.rows == 1);
    assert(dst.cols == src.cols);
    for (size_t row = 0; row < dst.rows; row++)
    {
        mat_elem_t sum = 0.0f;
        for (size_t col = 0; col < src.cols; col++)
        {
            sum += MAT_AT(src, row, col);
        }
        MAT_AT(dst, 0, row) = sum;
    }
}

Mat mat_sum_cols_alloc(Mat src)
{
    Mat dst = mat_alloc(1, src.cols);
    mat_sum_cols(dst, src);
    return dst;
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t row = 0; row < m.rows; row++)
    {
        printf("%*s    ", (int)padding, "");
        for (size_t col = 0; col < m.cols; col++)
        {
            printf("%f ", MAT_AT(m, row, col));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}
#endif // MAT_IMPLEMENTATION
#endif // MAT_H_
