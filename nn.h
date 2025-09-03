#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "mat.h"

#ifndef _NN_H_
#define _NN_H_

typedef struct
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;
    // 誤差逆伝播法用のメンバー
    Mat *zs;
    Mat *deltas;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

#define ARRAY_LEN(xs) (sizeof(xs) / sizeof((xs)[0]))

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_set_input(NN *nn, Mat m);
NN nn_clone_arch(NN nn);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_backprop_batch(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, Mat ti, Mat to, float rate);
void nn_free(NN nn);
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif //_NN_H_
