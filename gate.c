#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include "main.h"
#include "gate.h"
#include "mnist.h"

static t_mnist_data s_images = (t_mnist_data){0};
static t_mnist_data s_labels = (t_mnist_data){0};

static void read_images_and_labels()
{
	const char images_filename[] = "data/train-images-idx3-ubyte";
	const char labels_filename[] = "data/train-labels-idx1-ubyte";

	s_images = mnist_read_images(images_filename);
 	assert(s_images.adrs != NULL);
	printf("%s readed.\n", images_filename);

    s_labels = mnist_read_labels(labels_filename);
	assert(s_labels.adrs != NULL);
	printf("%s readed.\n", labels_filename);
}

// スレッドで実行されるカウンター関数
static void *counter_thread(void *arg)
{
	t_data  *data = (t_data *)arg;

	read_images_and_labels();
	printf("images: count=%ld,rows=%d,cols=%d\n", s_images.count, s_images.rows, s_images.cols);
	printf("labels: count=%ld\n", s_labels.count);
	
	pthread_mutex_lock(&data->mutex);
	data->image_adrs = s_images.adrs;
	data->cols = s_images.cols;
	data->rows = s_images.rows;
	data->counter = 0;
	pthread_mutex_unlock(&data->mutex);
	while (1) {
		printf("Counter: %ld\n", data->counter);
		sleep(1); // 1秒待機
		pthread_mutex_lock(&data->mutex);
		data->counter++;
		if (data->counter >= s_images.count)
			data->counter = 0;
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
