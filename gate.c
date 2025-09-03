#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include "main.h"
#include "gate.h"
#include "mnist.h"
#include "mat.h"
#include "nn.h"

typedef struct
{
	const char *name;
	Mat ti;
	Mat to;
	unsigned char *images;
	unsigned char *labels;
} Dataset;

const char train_images_path[] = "data/train-images-idx3-ubyte";
const char train_labels_path[] = "data/train-labels-idx1-ubyte";
const char t10k_images_path[] = "data/t10k-images-idx3-ubyte";
const char t10k_labels_path[] = "data/t10k-labels-idx1-ubyte";

Dataset dataset_mnist_load(const char *image_path, const char *label_path, const char *name)
{
	int images_num, labels_num;
	unsigned char *images = mnist_read_images(image_path, &images_num);
	unsigned char *labels = mnist_read_labels(label_path, &labels_num);

	assert(images != NULL && labels != NULL);
	assert(images_num == labels_num);

	size_t input_cols = 28 * 28;
	size_t output_cols = 10;

	Mat ti, to;
	ti = mat_alloc(images_num, input_cols);
	for (size_t row = 0; row < ti.rows; row++)
	{
		for (size_t col = 0; col < ti.cols; col++)
		{
			MAT_AT(ti, row, col) = (mat_elem_t)images[row * input_cols + col] / 255.f;
		}
	}

	to = mat_alloc(labels_num, output_cols);
	mat_zero(to);
	for (size_t row = 0; row < ti.rows; row++)
	{
		size_t label = (size_t)labels[row];
		MAT_AT(to, row, label) = 1.0f;
	}
	return (Dataset){.name = name, .ti = ti, .to = to, .images = images, .labels = labels};
}

Dataset dataset_mnist_train()
{
	return dataset_mnist_load(train_images_path, train_labels_path, "mnist_train");
}

Dataset dataset_mnist_test()
{
	return dataset_mnist_load(t10k_images_path, t10k_labels_path, "mnist_test");
}
//----------------------------------------
size_t count_correct_predictions(NN nn, Dataset ds, size_t sample_count)
{
	if (sample_count == 0 || sample_count > ds.ti.rows)
		sample_count = ds.ti.rows;

	size_t correct_predictions = 0;
	for (size_t iter = 0; iter < sample_count; iter++)
	{
		size_t idx = rand() % ds.ti.rows;

		nn_set_input(&nn, mat_row_view(ds.ti, idx));
		nn_forward(nn);

		size_t predicted = 0;
		for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
		{
			if (MAT_AT(NN_OUTPUT(nn), 0, j) > MAT_AT(NN_OUTPUT(nn), 0, predicted))
			{
				predicted = j;
			}
		}

		size_t true_label = ds.labels[idx];

		if (predicted == true_label)
		{
			correct_predictions++;
		}
	}

	return correct_predictions;
}

void print_cost(NN nn, Mat cost_ti, Mat cost_to, size_t epoch, size_t epochs)
{
	printf("epoch %zu/%zu, final cost = %f\n", (unsigned long)epoch, epochs, nn_cost(nn, cost_ti, cost_to));
}

void print_accuracy(NN nn, Dataset ds_test, size_t sample_count)
{
	size_t correct_prediction_count = count_correct_predictions(nn, ds_test, sample_count);
	float accuracy = (float)correct_prediction_count / sample_count;
	printf("Accuracy on %s: %.3f (%zu / %zu)\n", ds_test.name, accuracy, correct_prediction_count, sample_count);
}

//----------------------------------------
// スレッドで実行されるカウンター関数
static void *counter_thread(void *arg)
{
	t_data *data = (t_data *)arg;

	Dataset ds_train = dataset_mnist_load(train_images_path, train_labels_path, "mnist_train");
	Dataset ds_test = dataset_mnist_load(t10k_images_path, t10k_labels_path, "mnist_test");
	printf("images: count=%ld,%ld\n", ds_train.ti.rows, ds_test.ti.rows);
	printf("labels: count=%ld,%ld\n", ds_train.to.rows, ds_test.to.rows);

	pthread_mutex_lock(&data->mutex);
	data->mnist.counter = 0;
	for (size_t i = 0; i < 28 * 28; i++)
	{
		data->mnist.image_adrs[i] = (unsigned char)(ds_train.ti.es[data->mnist.counter * 28 * 28 + i] * 255);
	}
	pthread_mutex_unlock(&data->mutex);

	size_t arch[] = {ds_train.ti.cols, 100, 50, ds_train.to.cols};
	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	nn_rand(nn, -1, 1);

	float rate = 1e-2;
	size_t epochs = 25;
	size_t batch_size = 64;

	size_t sample_count = 1000;
	Mat cost_ti = mat_rows_view(ds_train.ti, 0, sample_count);
	Mat cost_to = mat_rows_view(ds_train.to, 0, sample_count);
	print_cost(nn, cost_ti, cost_to, 0, epochs);
	print_accuracy(nn, ds_test, sample_count);

	NN g = nn_clone_arch(nn);
	for (size_t i = 0; i < epochs; i++)
	{
		for (size_t j = 0; j < ds_train.ti.rows; j += batch_size)
		{
			size_t current_batch_size = (j + batch_size > ds_train.ti.rows) ? (ds_train.ti.rows - j) : batch_size;
			if (current_batch_size == 0)
				continue;

			Mat batch_ti = mat_rows_view(ds_train.ti, j, current_batch_size);
			Mat batch_to = mat_rows_view(ds_train.to, j, current_batch_size);

			nn_learn(nn, g, batch_ti, batch_to, rate);

			pthread_mutex_lock(&data->mutex);
			data->mnist.counter++;
			size_t cur = data->mnist.counter % ds_train.ti.rows;
			for (size_t i = 0; i < 28 * 28; i++)
			{
				data->mnist.image_adrs[i] = (unsigned char)(ds_train.ti.es[cur * 28 * 28 + i] * 256);
			}
			pthread_mutex_unlock(&data->mutex);
		}
		print_cost(nn, cost_ti, cost_to, i + 1, epochs);
		print_accuracy(nn, ds_test, sample_count);
	}
	nn_free(g);

	while (1)
	{
		pthread_mutex_lock(&data->mutex);
		size_t cur = data->mnist.counter;
		for (size_t i = 0; i < 28 * 28; i++)
		{
			data->mnist.image_adrs[i] = (unsigned char)(ds_train.ti.es[data->mnist.counter * 28 * 28 + i] * 256);
		}
		pthread_mutex_unlock(&data->mutex);
		printf("Counter: %ld\n", cur);
		sleep(1); // 1秒待機

		pthread_mutex_lock(&data->mutex);
		data->mnist.counter++;
		if (data->mnist.counter >= ds_train.ti.rows)
			data->mnist.counter = 0;
		pthread_mutex_unlock(&data->mutex);
	}
	return (NULL);
}

int gate_initialize(t_data *data)
{
	pthread_t thread_id;

	if (pthread_mutex_init(&data->mutex, NULL) != 0)
	{
		perror("pthread_mutex_init");
		return (-1);
	}
	// カウンタースレッドを作成し、data構造体を渡す
	if (pthread_create(&thread_id, NULL, counter_thread, data) != 0)
	{
		perror("pthread_create");
		pthread_mutex_destroy(&data->mutex);
		return (-1);
	}
	// スレッドをデタッチし、リソースが自動的に解放されるようにする
	pthread_detach(thread_id);

	return 0;
}
