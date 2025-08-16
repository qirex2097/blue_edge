#include <mlx.h>
#include <string.h>
#include <stdio.h>
#include "win.h"

int win_initialize(t_data *data)
{
	// minilibxの初期化
	data->mlx = mlx_init();
	if (!data->mlx)
		return (1);
	// 幅800, 高さ600のウィンドウを作成
	data->win = mlx_new_window(data->mlx, 800, 600, "10-second Window");
	if (!data->win)
		return (1);
	    
    return 0;
}

int render_next_frame(void* arg)
{
	t_data *data = (t_data*)arg;
	char buffer[256];

	if (data->counter != data->prev_counter) {
		// 画面をクリア
		mlx_clear_window(data->mlx, data->win);
		// 表示する文字列を作成
		sprintf(buffer, "Counter: %ld", data->counter);
		// 文字列をウィンドウに描画
		mlx_string_put(data->mlx, data->win, 10, 20, 0xFFFFFF, buffer); // 白色で表示
		printf("render_next_frame: %ld\n", data->counter);
		data->prev_counter = data->counter;
	}
	return 0;
}

int	key_press_hook(int keycode, t_data *data)
{
	if (keycode == 65307) // 65307 is the keycode for ESC
	{
		// ループを正常に終了させる
		mlx_loop_end(data->mlx);
	}
	return (0);
}

void win_loop(t_data *data)
{
	mlx_loop_hook(data->mlx, render_next_frame, data);
	mlx_key_hook(data->win, key_press_hook, data);
	// mlx_loop_endが呼ばれると、このループは終了する
	mlx_loop(data->mlx);
	// --- クリーンアップ処理 ---
	mlx_destroy_window(data->mlx, data->win);
	mlx_destroy_display(data->mlx);
}
