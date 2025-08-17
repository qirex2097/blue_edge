#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "main.h"
#include "gate.h"
#include "mnist.h"

static t_mnist_data s_images = (t_mnist_data){0};
static t_mnist_data s_labels = (t_mnist_data){0};

// スレッドで実行されるカウンター関数
void *counter_thread(void *arg)
{
	const char images_filename[] = "data/train-images-idx3-ubyte";
	const char labels_filename[] = "data/train-labels-idx1-ubyte";
	t_data  *data = (t_data *)arg;
	
	s_images = mnist_read_images(images_filename);
 	assert(s_images.adrs != NULL);
    s_labels = mnist_read_labels(labels_filename);
	assert(s_labels.adrs != NULL);

	printf("%s readed.\n", images_filename);
	printf("%s readed.\n", labels_filename);

	data->counter = 0;
	while (1) {
		// printf("Counter: %ld\n", data->counter);
		sleep(1); // 1秒待機
		data->counter++;
	}
	return (NULL);
}

int gate_initialize(t_data *data)
{
	pthread_t   thread_id;

	// カウンタースレッドを作成し、data構造体を渡す
	if (pthread_create(&thread_id, NULL, counter_thread, data) != 0)
	{
		perror("pthread_create");
		return (-1);
	}
	// スレッドをデタッチし、リソースが自動的に解放されるようにする
	pthread_detach(thread_id);

	return 0;
}
