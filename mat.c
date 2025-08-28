#include <math.h>
#include <time.h>
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "mnist.h"

Mat mat_slice_view(Mat src, size_t start, size_t kazu)
{
    if (start + kazu >= src.rows)
        kazu = src.rows - start;
    return (Mat){.rows = kazu, .cols = src.cols, .data = &src.data[start * src.cols]};
}

//----------------------------------------
typedef struct
{
    Mat W, b;
    Mat x;
    Mat dW, db;
} Affine;

Affine affine_init(size_t input, size_t output)
{
    Affine layer;
    layer.W = mat_alloc(input, output);
    mat_rand(layer.W, -1.0, 1.0);
    layer.b = mat_alloc(1, output);
    mat_zero(layer.b);
    layer.dW = mat_alloc(input, output);
    layer.db = mat_alloc(1, output);
    layer.x = (Mat){0};
    return layer;
}

void affine_free(Affine *layer)
{
    mat_free(&layer->W);
    mat_free(&layer->b);
    mat_free(&layer->dW);
    mat_free(&layer->db);
    *layer = (Affine){0};
}

Mat affine_forward(Affine *layer, Mat x)
{
    layer->x = x;
    Mat y = mat_dot_alloc(x, layer->W);   // y = x*W + b
    mat_add_rowwise_inplace(y, layer->b); //
    return y;
}

Mat affine_backward(Affine *layer, Mat dy)
{
    Mat Wt = mat_transpose_alloc(layer->W);
    Mat dx = mat_dot_alloc(dy, Wt); // dx = dy * W^T
    mat_free(&Wt);

    Mat xt = mat_transpose_alloc(layer->x);
    mat_dot(layer->dW, xt, dy); // dW = x^T * dy
    mat_free(&xt);

    mat_sum_cols(layer->db, dy); // db = sum(dy)

    return dx;
}

void affine_update(Affine *layer, float lr)
{
    // W = W - lr * dW
    for (size_t i = 0; i < layer->W.rows; i++)
    {
        for (size_t j = 0; j < layer->W.cols; j++)
        {
            layer->W.data[i * layer->W.cols + j] -= lr * layer->dW.data[i * layer->dW.cols + j];
        }
    }

    // b = b - lr * db
    for (size_t j = 0; j < layer->b.cols; j++)
    {
        layer->b.data[j] -= lr * layer->db.data[j];
    }
}
//----------------------------------------
typedef struct
{
    Mat mask;
} ReLU;

ReLU ReLU_init(size_t row, size_t col)
{
    ReLU layer;
    layer.mask = mat_alloc(row, col);
    return layer;
}

void ReLU_free(ReLU *layer)
{
    mat_free(&layer->mask);
    *layer = (ReLU){0};
}

Mat ReLU_forward(ReLU *layer, Mat x)
{
    Mat y = mat_alloc(x.rows, x.cols);
    for (size_t row = 0; row < x.rows; row++)
    {
        for (size_t col = 0; col < x.cols; col++)
        {
            mat_elem_t val = MAT_AT(x, row, col);
            if (val > 0)
            {
                MAT_AT(y, row, col) = val;
                MAT_AT(layer->mask, row, col) = 1.0f;
            }
            else
            {
                MAT_AT(y, row, col) = 0;
                MAT_AT(layer->mask, row, col) = 0.0f;
            }
        }
    }
    return y;
}

Mat ReLU_backward(ReLU *layer, Mat dy)
{
    Mat dx = mat_alloc(dy.rows, dy.cols);
    for (size_t row = 0; row < dx.rows; row++)
    {
        for (size_t col = 0; col < dx.cols; col++)
        {
            if (MAT_AT(layer->mask, row, col))
                MAT_AT(dx, row, col) = MAT_AT(dy, row, col);
            else
                MAT_AT(dx, row, col) = 0;
        }
    }
    return dx;
}
//----------------------------------------
Mat softmax(Mat src)
{
    Mat dst = mat_alloc(src.rows, src.cols);
    for (size_t row = 0; row < src.rows; row++)
    {
        mat_elem_t max_val = MAT_AT(src, row, 0);
        for (size_t col = 1; col < src.cols; col++)
        {
            if (MAT_AT(src, row, col) > max_val)
                max_val = MAT_AT(src, row, col);
        }

        mat_elem_t sum_exp = 0;
        for (size_t col = 0; col < src.cols; col++)
        {
            mat_elem_t e = exp(MAT_AT(src, row, col) - max_val);
            MAT_AT(dst, row, col) = e;
            sum_exp += e;
        }

        for (size_t col = 0; col < src.cols; col++)
        {
            MAT_AT(dst, row, col) /= sum_exp;
        }
    }
    return dst;
}

Mat softmax_cross_entropy_backward(Mat y, Mat t)
{
    Mat dx = mat_alloc(y.rows, y.cols);
    for (size_t row = 0; row < y.rows; row++)
    {
        for (size_t col = 0; col < y.cols; col++)
        {
            MAT_AT(dx, row, col) = (MAT_AT(y, row, col) - MAT_AT(t, row, col)) / y.rows;
        }
    }
    return dx;
}

mat_elem_t cross_entropy(Mat y, Mat t)
{
    mat_elem_t loss = 0;
    for (size_t row = 0; row < y.rows; row++)
    {
        for (size_t col = 0; col < y.cols; col++)
        {
            if (MAT_AT(t, row, col) == 1.0)
            {
                loss -= log(MAT_AT(y, row, col) + 1e-7);
            }
        }
    }
    return loss / y.rows;
}
//----------------------------------------
typedef struct
{
    // パラメータ初期化
    Affine l1;
    ReLU r1;
    Affine l2;
    ReLU r2;
    Affine l3;

} NN;

NN NN_init(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t output_size, size_t batch_size)
{
    NN nn;
    nn.l1 = affine_init(input_size, hidden1_size);
    nn.r1 = ReLU_init(batch_size, hidden1_size);
    nn.l2 = affine_init(hidden1_size, hidden2_size);
    nn.r2 = ReLU_init(batch_size, hidden2_size);
    nn.l3 = affine_init(hidden2_size, output_size);
    return nn;
}
void NN_free(NN *nn)
{
    affine_free(&nn->l1);
    ReLU_free(&nn->r1);
    affine_free(&nn->l2);
    ReLU_free(&nn->r2);
    affine_free(&nn->l3);
}

Mat NN_forward(NN *nn, Mat x_batch)
{
    Mat a1 = affine_forward(&nn->l1, x_batch);
    Mat z1 = ReLU_forward(&nn->r1, a1);

    Mat a2 = affine_forward(&nn->l2, z1);
    Mat z2 = ReLU_forward(&nn->r2, a2);

    Mat a3 = affine_forward(&nn->l3, z2);
    Mat y = softmax(a3);

    mat_free(&a1);
    mat_free(&z1);
    mat_free(&a2);
    mat_free(&z2);
    mat_free(&a3);

    return y;
}

void NN_backward(NN *nn, Mat y, Mat t_batch)
{
    Mat dy = softmax_cross_entropy_backward(y, t_batch);
    Mat dz2 = affine_backward(&nn->l3, dy);
    Mat da2 = ReLU_backward(&nn->r2, dz2);

    Mat dz1 = affine_backward(&nn->l2, da2);
    Mat da1 = ReLU_backward(&nn->r1, dz1);

    affine_backward(&nn->l1, da1);

    mat_free(&dy);
    mat_free(&dz2);
    mat_free(&da2);
    mat_free(&dz1);
    mat_free(&da1);
}

void NN_update(NN *nn, float learning_rate)
{
    affine_update(&nn->l1, learning_rate);
    affine_update(&nn->l2, learning_rate);
    affine_update(&nn->l3, learning_rate);
}
//----------------------------------------
Mat read_images(const char *filename)
{
    t_mnist_data mnist_data = mnist_read_images(filename);
    assert(mnist_data.adrs != NULL);
    printf("%s readed. image num = %zu\n", filename, mnist_data.count);

    size_t images_num = mnist_data.count;
    size_t input_cols = 28 * 28;
    Mat ti = mat_alloc(images_num, input_cols);
    for (size_t row = 0; row < images_num; row++)
    {
        for (size_t col = 0; col < input_cols; col++)
        {
            MAT_AT(ti, row, col) = (mat_elem_t)mnist_data.adrs[row * input_cols + col] / 255.f;
        }
    }
    free(mnist_data.adrs);
    return ti;
}

Mat read_labels(const char *filename)
{
    t_mnist_data mnist_data = mnist_read_labels(filename);
    assert(mnist_data.adrs != NULL);
    printf("%s readed. label num = %zu\n", filename, mnist_data.count);

    size_t labels_num = mnist_data.count;
    size_t output_cols = 10;
    Mat to = mat_alloc(labels_num, output_cols);
    mat_zero(to);
    for (size_t row = 0; row < labels_num; row++)
    {
        MAT_AT(to, row, mnist_data.adrs[row]) = 1;
    }
    free(mnist_data.adrs);
    return to;
}

// 指定されたインデックスの画像とラベルを表示する
void print_image(Mat ti, Mat to, size_t index)
{
    mat_elem_t *images = &MAT_AT(ti, index, 0);
    size_t label = 0;
    for (size_t col = 0; col < to.cols; col++)
    {
        if (MAT_AT(to, index, col) == 1.0f)
            label = col;
    }
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            mat_elem_t pixel = images[i * 28 + j];
            printf("%c", pixel > 0.5f ? '#' : '.');
        }
        printf("\n");
    }
    printf("Image #%zu, Label: %zu\n", index, label);
}
//----------------------------------------
float calculate_accuracy(Mat predictions, Mat targets);
void mnist()
{
    srand(time(0));
    // MNIST読み込み
    Mat train_x = read_images("data/train-images-idx3-ubyte");
    Mat train_t = read_labels("data/train-labels-idx1-ubyte");

    size_t input_size = 784;
    size_t hidden1_size = 128;
    size_t hidden2_size = 64;
    size_t output_size = 10;
    size_t batch_size = 100;
    size_t epochs = 10;
    float learning_rate = 0.001f;

    // パラメータ初期化
    NN nn = NN_init(input_size, hidden1_size, hidden2_size, output_size, batch_size);

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        for (size_t i = 0; i < train_x.rows; i += batch_size)
        {
            // ミニバッチ作成
            Mat x_batch = mat_slice_view(train_x, i, batch_size);
            Mat t_batch = mat_slice_view(train_t, i, batch_size);

            // --- forward ---
            Mat y = NN_forward(&nn, x_batch);
            mat_elem_t loss = cross_entropy(y, t_batch);

            // --- backward ---
            NN_backward(&nn, y, t_batch);
            // --- 更新 ---
            NN_update(&nn, learning_rate);

            if (i % 5000 == 0)
            {
                float batch_accuracy = calculate_accuracy(y, t_batch);
                printf("epoch %zu, iter %zu, loss = %f, batch_acc = %f\n", epoch, i, loss, batch_accuracy);
            }

            // メモリ解放
            mat_free(&y);
            // x_batch and t_batch are views, do not free them.
        }
    }

    NN_free(&nn);
    mat_free(&train_x);
    mat_free(&train_t);
}

// 正解率を計算する関数
float calculate_accuracy(Mat predictions, Mat targets)
{
    size_t correct = 0;

    for (size_t row = 0; row < predictions.rows; row++)
    {
        // 予測：最大値のインデックスを見つける
        size_t pred_class = 0;
        mat_elem_t max_pred = MAT_AT(predictions, row, 0);
        for (size_t col = 1; col < predictions.cols; col++)
        {
            if (MAT_AT(predictions, row, col) > max_pred)
            {
                max_pred = MAT_AT(predictions, row, col);
                pred_class = col;
            }
        }

        // 正解：1が立っているインデックスを見つける
        size_t true_class = 0;
        for (size_t col = 0; col < targets.cols; col++)
        {
            if (MAT_AT(targets, row, col) == 1.0f)
            {
                true_class = col;
                break;
            }
        }

        if (pred_class == true_class)
        {
            correct++;
        }
    }

    return (float)correct / predictions.rows;
}

// より効率的な版（テスト用データセット全体の正解率）
float evaluate_accuracy(Mat test_x, Mat test_t, Affine *l1, ReLU *r1, Affine *l2, ReLU *r2, Affine *l3)
{
    size_t batch_size = 100; // 評価用のバッチサイズ
    size_t total_correct = 0;
    size_t total_samples = 0;

    for (size_t i = 0; i < test_x.rows; i += batch_size)
    {
        size_t current_batch_size = (i + batch_size > test_x.rows) ? (test_x.rows - i) : batch_size;

        Mat x_batch = mat_slice_view(test_x, i, current_batch_size);
        Mat t_batch = mat_slice_view(test_t, i, current_batch_size);

        // Forward pass
        Mat a1 = affine_forward(l1, x_batch);
        Mat z1 = ReLU_forward(r1, a1);
        Mat a2 = affine_forward(l2, z1);
        Mat z2 = ReLU_forward(r2, a2);
        Mat a3 = affine_forward(l3, z2);
        Mat y = softmax(a3);

        // 正解数をカウント
        for (size_t row = 0; row < current_batch_size; row++)
        {
            size_t pred_class = 0;
            mat_elem_t max_pred = MAT_AT(y, row, 0);
            for (size_t col = 1; col < y.cols; col++)
            {
                if (MAT_AT(y, row, col) > max_pred)
                {
                    max_pred = MAT_AT(y, row, col);
                    pred_class = col;
                }
            }

            size_t true_class = 0;
            for (size_t col = 0; col < t_batch.cols; col++)
            {
                if (MAT_AT(t_batch, row, col) == 1.0f)
                {
                    true_class = col;
                    break;
                }
            }

            if (pred_class == true_class)
            {
                total_correct++;
            }
        }

        total_samples += current_batch_size;

        // メモリ解放
        mat_free(&a1);
        mat_free(&z1);
        mat_free(&a2);
        mat_free(&z2);
        mat_free(&a3);
        mat_free(&y);
    }

    return (float)total_correct / total_samples;
}
#ifdef MAT_STANDALONE
int main(void)
{
    mnist();
    return 0;
}
#endif
