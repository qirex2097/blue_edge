#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "main.h"
#include "gate.h"

// スレッドで実行されるカウンター関数
void    *counter_thread(void *arg)
{
	t_data  *data = (t_data *)arg;

	data->counter = 0;
	while (1) {
		printf("Counter: %ld\n", data->counter);
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
		return (1);
	}
	// スレッドをデタッチし、リソースが自動的に解放されるようにする
	pthread_detach(thread_id);

	return 0;
}
