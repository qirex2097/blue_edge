#include <mlx.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "win.h"

static int s_debug_y = 10;
void debug_reset_lines(void) { s_debug_y = 10; }

void debug_print(t_data *data, const char *line, ...)
{
	char buffer[256];
	
	// 表示する文字列を作成
	va_list ap;
	va_start(ap, line);
	vsnprintf(buffer, sizeof(buffer), line, ap);
	va_end(ap);

	// 文字列をウィンドウに描画
	mlx_string_put(data->mlx, data->win, 20, s_debug_y, 0xFFFFFF, buffer); // 白色で表示

	s_debug_y += 10;
}

int win_initialize(t_data *data, int width, int height)
{
	// minilibxの初期化
	data->mlx = mlx_init();
	if (!data->mlx) {
		perror("mlx_init");
		return (1);
	}

	// 幅 width, 高さ height のウィンドウを作成
	data->win = mlx_new_window(data->mlx, width, height, "10-second Window");
	if (!data->win) {
		perror("mlx_new_window");
		return (2);
	}
	    
    return 0;
}

int render_next_frame(void* arg)
{
	t_data *data = (t_data*)arg;

	// 画面をクリア
	mlx_clear_window(data->mlx, data->win);
	debug_reset_lines();

	debug_print(data, "adrs: 0x%x", data->image_adrs);

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
