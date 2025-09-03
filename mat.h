#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef MAT_H_
#define MAT_H_

typedef float mat_elem_t;

typedef struct
{
    size_t rows;
    size_t cols;
    mat_elem_t *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]
#define MAT_FREE(m) (m) = mat_free((m))
#define MAT_RESIZE(m, i, j) (m) = mat_resize((m), i, j)

mat_elem_t rand_float(void);
mat_elem_t sigmoidf(mat_elem_t x);
mat_elem_t dsigmoidf(mat_elem_t x);

Mat mat_alloc(size_t rows, size_t cols);
Mat mat_resize(Mat m, size_t rows, size_t cols);
Mat mat_free(Mat m);
void mat_zero(Mat m);
void mat_fill(Mat m, mat_elem_t x);
void mat_rand(Mat m, mat_elem_t low, mat_elem_t high);
Mat mat_row_view(Mat m, size_t row);
Mat mat_rows_view(Mat m, size_t i, size_t rows);
void mat_copy_inline(Mat dst, Mat src);
Mat mat_dot(Mat a, Mat b);
Mat mat_transpose(Mat a);
void mat_dot_transposeA_inline(Mat C, Mat A, Mat B);
void mat_dot_inline(Mat dst, Mat a, Mat b);
void mat_sum_inline(Mat dst, Mat a);
void mat_sum_bias_inline(Mat dst, Mat bias);
void mat_scalar_div_inline(Mat dst, float scalar);
void mat_sig(Mat m);
void mat_dsig(Mat dst, Mat src);
void mat_softmax(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

#endif // MAT_H_