#ifndef NN_H_
#define NN_H_
#include "mat.h"

typedef struct
{
    Mat W, b;
    Mat x;
    Mat dW, db;
} Affine;

typedef struct
{
    Mat mask;
} ReLU;

typedef struct
{
    // パラメータ初期化
    Affine l1;
    ReLU r1;
    Affine l2;
    ReLU r2;
    Affine l3;
} NN;

NN NN_init(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t output_size, size_t batch_size);
void NN_free(NN *nn);
Mat NN_forward(NN *nn, Mat x_batch);
void NN_backward(NN *nn, Mat y, Mat t_batch);
void NN_update(NN *nn, float learning_rate);

mat_elem_t cross_entropy(Mat y, Mat t);

Mat read_images(const char *filename);
Mat read_labels(const char *filename);
void print_image(Mat ti, Mat to, size_t index);
float calculate_accuracy(Mat predictions, Mat targets);

#endif//NN_H_