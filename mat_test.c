#include "mat.h"

void test_mat_alloc()
{
    printf("Testing mat_alloc...\n");
    Mat m = mat_alloc(2, 3);
    assert(m.rows == 2);
    assert(m.cols == 3);
    assert(m.data);
    mat_free(&m);
    assert(m.rows == 0);
    assert(m.cols == 0);
    assert(!m.data);
    printf("mat_alloc test passed.\n");
}

void test_mat_zero()
{
    printf("Testing mat_zero...\n");
    Mat m = mat_alloc(2, 3);
    mat_rand(m, 0, 1);
    mat_zero(m);
    for (size_t i = 0; i < m.rows * m.cols; i++)
    {
        assert(m.data[i] == 0);
    }
    mat_free(&m);
    printf("mat_zero test passed.\n");
}

void test_mat_rand()
{
    printf("Testing mat_rand...\n");
    Mat m = mat_alloc(2, 3);
    mat_rand(m, 5, 10);
    for (size_t i = 0; i < m.rows * m.cols; i++)
    {
        assert(m.data[i] >= 5 && m.data[i] <= 10);
    }
    mat_free(&m);
    printf("mat_rand test passed.\n");
}

void test_mat_add_inplace()
{
    printf("Testing mat_add_inplace...\n");
    Mat a = mat_alloc(2, 2);
    mat_rand(a, 0, 1);
    Mat b = mat_alloc(2, 2);
    mat_rand(b, 0, 1);

    Mat c = mat_alloc(2, 2);
    for (size_t i = 0; i < a.rows * a.cols; i++)
    {
        c.data[i] = a.data[i] + b.data[i];
    }

    mat_add_inplace(a, b);

    for (size_t i = 0; i < a.rows * a.cols; i++)
    {
        assert(a.data[i] == c.data[i]);
    }

    mat_free(&a);
    mat_free(&b);
    mat_free(&c);
    printf("mat_add_inplace test passed.\n");
}

void test_mat_dot()
{
    printf("Testing mat_dot...\n");
    Mat a = mat_alloc(2, 3);
    mat_rand(a, 0, 1);
    Mat b = mat_alloc(3, 2);
    mat_rand(b, 0, 1);
    Mat c = mat_alloc(2, 2);
    mat_dot(c, a, b);

    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
            mat_elem_t sum = 0;
            for (size_t k = 0; k < a.cols; k++)
            {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            assert(MAT_AT(c, i, j) == sum);
        }
    }

    mat_free(&a);
    mat_free(&b);
    mat_free(&c);
    printf("mat_dot test passed.\n");
}

void test_mat_transpose()
{
    printf("Testing mat_transpose...\n");
    Mat a = mat_alloc(2, 3);
    mat_rand(a, 0, 1);
    Mat b = mat_alloc(3, 2);
    mat_transpose(b, a);

    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            assert(MAT_AT(a, i, j) == MAT_AT(b, j, i));
        }
    }

    mat_free(&a);
    mat_free(&b);
    printf("mat_transpose test passed.\n");
}
