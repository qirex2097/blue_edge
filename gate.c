#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include "main.h"
#include "gate.h"
#include "mnist.h"
#include "mat.h"
#include "nn.h"

typedef struct {
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
	for (size_t row = 0; row < ti.rows; row++) {
		for (size_t col = 0; col < ti.cols; col++) {
			MAT_AT(ti, row, col) = (mat_elem_t)images[row * input_cols + col] / 255.f;
		}
	}
	
	to = mat_alloc(labels_num, output_cols);
	mat_zero(to);
	for (size_t row = 0; row < ti.rows; row++) {
		size_t label = (size_t)labels[row];
		MAT_AT(to, row, label) = 1.0f;
	}
	return (Dataset){ .name = name, .ti = ti, .to = to, .images = images, .labels = labels };
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
// スレッドで実行されるカウンター関数
static void *counter_thread(void *arg)
{
	t_data  *data = (t_data *)arg;

	Dataset dataset = dataset_mnist_load(train_images_path, train_labels_path, "minst_train");
	printf("images: count=%ld\n", dataset.ti.rows);
	printf("labels: count=%ld\n", dataset.to.rows);
	
	pthread_mutex_lock(&data->mutex);
	data->mnist.counter = 0;
	for (size_t i = 0; i < 28 * 28; i++) {
		data->mnist.image_adrs[i] = (unsigned char)(dataset.ti.es[data->mnist.counter * 28 * 28 + i] * 256);
	}
	pthread_mutex_unlock(&data->mutex);
	while (1) {
		pthread_mutex_lock(&data->mutex);
		size_t cur = data->mnist.counter;
		for (size_t i = 0; i < 28 * 28; i++) {
			data->mnist.image_adrs[i] = (unsigned char)(dataset.ti.es[data->mnist.counter * 28 * 28 + i] * 256);
		}
		pthread_mutex_unlock(&data->mutex);
		printf("Counter: %ld\n", cur);
		sleep(1); // 1秒待機

		pthread_mutex_lock(&data->mutex);
		data->mnist.counter++;
		if (data->mnist.counter >= dataset.ti.rows)
			data->mnist.counter = 0;
		pthread_mutex_unlock(&data->mutex);
	}
	return (NULL);
}

int gate_initialize(t_data *data)
{
	pthread_t   thread_id;

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
