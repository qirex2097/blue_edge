#include <mlx.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

// mlxポインタをスレッドに渡すための構造体
typedef struct s_data
{
	void	*mlx;
	void	*win;
	size_t	counter;
	size_t  prev_counter;
}			t_data;

// スレッドで実行されるカウンター関数
void	*counter_thread(void *arg)
{
	t_data	*data = (t_data *)arg;

	data->counter = 0;
	while (1) {
		printf("Counter: %ld\n", data->counter);
		sleep(1); // 1秒待機
		data->counter++;
	}
	return (NULL);
}

int render_next_frame(void* arg)
{
	t_data *data = (t_data*)arg;
	if (data->counter != data->prev_counter) {
		printf("render_next_frame: %ld\n", data->counter);
		data->prev_counter = data->counter;
	}
	return 0;
}

int	main(void)
{
	t_data		data;
	pthread_t	thread_id;

	// minilibxの初期化
	data.mlx = mlx_init();
	if (!data.mlx)
		return (1);
	// 幅800, 高さ600のウィンドウを作成
	data.win = mlx_new_window(data.mlx, 800, 600, "10-second Window");
	if (!data.win)
		return (1);
	// カウンタースレッドを作成し、data構造体を渡す
	if (pthread_create(&thread_id, NULL, counter_thread, &data) != 0)
	{
		perror("pthread_create");
		return (1);
	}
	// スレッドの終了を待つ
	pthread_detach(thread_id);
	// イベントループ開始
	mlx_loop_hook(data.mlx, render_next_frame, (void*)&data);
	// counter_threadでmlx_loop_endが呼ばれると、このループは終了する
	mlx_loop(data.mlx);
	// mlxのリソースを解放
	mlx_destroy_display(data.mlx);
	printf("Program finished.\n");
	return (0);
}