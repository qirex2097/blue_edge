#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include "main.h"
#include "gate.h"
#include "mnist.h"
#include "mat.h"
#include "nn.h"

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
	data->mnist.image_adrs = s_images.adrs;
	data->mnist.cols = s_images.cols;
	data->mnist.rows = s_images.rows;
	data->mnist.counter = 0;
	pthread_mutex_unlock(&data->mutex);
	while (1) {
		pthread_mutex_lock(&data->mutex);
		size_t cur = data->mnist.counter;
		pthread_mutex_unlock(&data->mutex);
		printf("Counter: %ld\n", cur);
		sleep(1); // 1秒待機

		pthread_mutex_lock(&data->mutex);
		data->mnist.counter++;
		if (data->mnist.counter >= s_images.count)
			data->mnist.counter = 0;
		pthread_mutex_unlock(&data->mutex);
	}
	return (NULL);
}

static void mnist_thread(void *arg);

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
	
	(void)mnist_thread;

	return 0;
}

static void mnist_thread(void *arg)
{
	t_data *data = (t_data *)arg;
	(void)data;

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

            if (i % 1000 == 0)
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
