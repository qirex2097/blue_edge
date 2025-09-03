#include "nn.h"

void nn_debug(NN *nn)
{
    printf("as[%zu]: rows=%zu,cols=%zu\n", (size_t)0, nn->as[0].rows, nn->as[0].cols);
    for (size_t i = 1; i < nn->count + 1; i++)
    {
        printf("ws[%zu]: rows=%zu,cols=%zu\n", i - 1, nn->ws[i - 1].rows, nn->ws[i - 1].cols);
        printf("bs[%zu]: rows=%zu,cols=%zu\n", i - 1, nn->bs[i - 1].rows, nn->bs[i - 1].cols);
        printf("zs[%zu]: rows=%zu,cols=%zu\n", i - 1, nn->zs[i - 1].rows, nn->zs[i - 1].cols);
        printf("deltas[%zu]: rows=%zu,cols=%zu\n", i - 1, nn->deltas[i - 1].rows, nn->deltas[i - 1].cols);
        printf("as[%zu]: rows=%zu,cols=%zu\n", i, nn->as[i].rows, nn->as[i].cols);
    }
}

/*
 * size_t arch[] = { 2, 2, 1 };
 * NN nn = nn_alloc(arch, ARRAY_LEN(arch));
 */
NN nn_alloc(size_t *arch, size_t arch_count)
{
    assert(arch && arch_count >= 2 && "architecture must have input and output layer");

    NN nn;
    nn.count = arch_count - 1;
    nn.ws = malloc(sizeof(*nn.ws) * nn.count);
    assert(nn.ws != NULL);
    nn.bs = malloc(sizeof(*nn.bs) * nn.count);
    assert(nn.bs != NULL);
    nn.as = malloc(sizeof(*nn.as) * arch_count);
    assert(nn.as != NULL);
    nn.zs = malloc(sizeof(*nn.zs) * nn.count);
    assert(nn.zs != NULL);
    nn.deltas = malloc(sizeof(*nn.deltas) * nn.count);
    assert(nn.deltas != NULL);

    assert(arch[0] > 0);
    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; i++)
    {
        assert(arch[i] > 0 && "layer must be > 0");
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
        nn.zs[i - 1] = mat_alloc(1, arch[i]);
        nn.deltas[i - 1] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_set_input(NN *nn, Mat m)
{
    assert(nn && nn->count > 0);
    assert(m.cols == nn->ws[0].rows && "input width must match first-layer weight rows");
    MAT_RESIZE(NN_INPUT(*nn), m.rows, m.cols);
    mat_copy_inline(NN_INPUT(*nn), m);
    for (size_t i = 0; i < nn->count; i++)
    {
        MAT_RESIZE(nn->as[i + 1], m.rows, nn->ws[i].cols);
        MAT_RESIZE(nn->zs[i], m.rows, nn->ws[i].cols);
        MAT_RESIZE(nn->deltas[i], m.rows, nn->ws[i].cols);
    }
    // MAT_RESIZE(NN_OUTPUT(*nn), m.rows, nn->ws[nn->count - 1].cols);
}

NN nn_clone_arch(NN nn)
{
    size_t arch_count = nn.count + 1;
    size_t *arch = malloc(sizeof(*arch) * arch_count);
    assert(arch != NULL);
    for (size_t i = 0; i < arch_count; ++i)
    {
        arch[i] = nn.as[i].cols;
    }
    NN result = nn_alloc(arch, arch_count);
    free(arch);
    return result;
}

void nn_free(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        MAT_FREE(nn.ws[i]);
        MAT_FREE(nn.bs[i]);
        MAT_FREE(nn.zs[i]);
        MAT_FREE(nn.deltas[i]);
    }
    for (size_t i = 0; i < nn.count + 1; i++)
    {
        MAT_FREE(nn.as[i]);
    }

    free(nn.ws);
    free(nn.bs);
    free(nn.as);
    free(nn.zs);
    free(nn.deltas);
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        // z=a*w+b を計算し、zsに保存
        mat_dot_inline(nn.zs[i], nn.as[i], nn.ws[i]);
        mat_sum_bias_inline(nn.zs[i], nn.bs[i]);
        // a=sigmoid(z) を計算
        mat_copy_inline(nn.as[i + 1], nn.zs[i]);
        if (i + 1 == nn.count)
        {
            mat_softmax(nn.as[i + 1]);
        }
        else
        {
            mat_sig(nn.as[i + 1]);
        }
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0.f;
    for (size_t i = 0; i < n; ++i)
    {
        Mat x = mat_row_view(ti, i);
        Mat y = mat_row_view(to, i);

        // mat_copy_inline(NN_INPUT(nn), x);
        nn_set_input(&nn, x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            // 交差エントロピー誤差を計算
            // c += -MAT_AT(y, 0, j) * logf(MAT_AT(NN_OUTPUT(nn), 0, j));
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);
            if (p < 1e-12f) p = 1e-12f;
            c += -MAT_AT(y, 0, j) * logf(p);
        }
    }

    return c / n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    for (size_t i = 0; i < nn.count; i++)
    {
        mat_zero(g.ws[i]);
        mat_zero(g.bs[i]);
    }

    for (size_t sample = 0; sample < n; sample++)
    {
        Mat x = mat_row_view(ti, sample);
        Mat y = mat_row_view(to, sample);

        // Forward pass
        mat_copy_inline(NN_INPUT(nn), x);
        nn_forward(nn);

        // Backward pass
        // 出力層の誤差を計算 (交差エントロピー + Softmax)
        size_t output_layer = nn.count - 1;
        for (size_t j = 0; j < nn.deltas[output_layer].cols; j++)
        {
            float output = MAT_AT(NN_OUTPUT(nn), 0, j);
            float target = MAT_AT(y, 0, j);
            MAT_AT(nn.deltas[output_layer], 0, j) = output - target;
        }

        // 隠れ層の誤差を逆伝播
        for (int layer = (int)nn.count - 2; layer >= 0; layer--)
        {
            for (size_t j = 0; j < nn.deltas[layer].cols; j++)
            {
                float sum = 0.f;
                for (size_t k = 0; k < nn.deltas[layer + 1].cols; k++)
                {
                    sum += MAT_AT(nn.deltas[layer + 1], 0, k) * MAT_AT(nn.ws[layer + 1], j, k);
                }
                float a = MAT_AT(nn.as[layer + 1], 0, j);
                MAT_AT(nn.deltas[layer], 0, j) = sum * a * (1.f - a);
            }
        }

        // 重みとバイアスの勾配を計算
        for (size_t layer = 0; layer < nn.count; layer++)
        {
            for (size_t i = 0; i < nn.ws[layer].rows; i++)
            {
                for (size_t j = 0; j < nn.ws[layer].cols; j++)
                {
                    float a = MAT_AT(nn.as[layer], 0, i);
                    float delta = MAT_AT(nn.deltas[layer], 0, j);
                    MAT_AT(g.ws[layer], i, j) += a * delta;
                }
            }

            for (size_t j = 0; j < nn.ws[layer].cols; j++)
            {
                float delta = MAT_AT(nn.deltas[layer], 0, j);
                MAT_AT(g.bs[layer], 0, j) += delta;
            }
        }
    }

    // 平均をとる
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < g.ws[i].rows; j++)
        {
            for (size_t k = 0; k < g.ws[i].cols; k++)
            {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].cols; j++)
        {
            MAT_AT(g.bs[i], 0, j) /= n;
        }
    }
}

void nn_backprop_batch(NN nn, NN g, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    nn_set_input(&nn, ti);

    // ---- Forward ----
    nn_forward(nn); // バッチ全体をforwardする版が必要

    // ---- Backward ----
    size_t L = nn.count - 1;

    // 出力層の誤差 Δ = y - t
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < nn.deltas[L].cols; j++)
        {
            float output = MAT_AT(NN_OUTPUT(nn), i, j);
            float target = MAT_AT(to, i, j);
            MAT_AT(nn.deltas[L], i, j) = output - target;
        }
    }

    // 隠れ層の誤差
    for (int layer = (int)L - 1; layer >= 0; layer--)
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < nn.deltas[layer].cols; j++)
            {
                float sum = 0.f;
                for (size_t k = 0; k < nn.deltas[layer + 1].cols; k++)
                {
                    sum += MAT_AT(nn.deltas[layer + 1], i, k) * MAT_AT(nn.ws[layer + 1], j, k);
                }
                float a = MAT_AT(nn.as[layer + 1], i, j);
                MAT_AT(nn.deltas[layer], i, j) = sum * a * (1.f - a);
            }
        }
    }

    // ---- 勾配計算 (行列積でまとめて処理) ----
    for (size_t layer = 0; layer < nn.count; layer++)
    {
        // g.ws[layer] = A^T * Δ / n
        mat_dot_transposeA_inline(g.ws[layer], nn.as[layer], nn.deltas[layer]);

        // g.bs[layer] = 平均Δ
        // → 行ごとの平均を取る
        for (size_t j = 0; j < g.bs[layer].cols; j++)
        {
            float sum = 0.f;
            for (size_t i = 0; i < n; i++)
            {
                sum += MAT_AT(nn.deltas[layer], i, j);
            }
            MAT_AT(g.bs[layer], 0, j) = sum / n;
        }
    }
}

void nn_update(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void nn_learn(NN nn, NN g, Mat ti, Mat to, float rate)
{
    // nn_backprop(nn, g, ti, to);
    nn_backprop_batch(nn, g, ti, to);
    nn_update(nn, g, rate);
}
