#include <mlx.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "main.h"

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
	mlx_string_put(data->mlx_ptr, data->win_ptr, 20, s_debug_y, 0xFFFFFF, buffer); // 白色で表示

	s_debug_y += 10;
}

void my_img_clear(t_img *img)
{
	assert(img);
	assert(img->adrs);
	memset(img->adrs, 0, img->w * img->h * (img->bits_per_pixel / 8));
}

void my_img_pixel_put(t_img *img, int x, int y, int color)
{
	if (!img || !img->adrs || x < 0 || img->w <= x || y < 0 || img->h <= y) return;
	char *dst = img->adrs + (y * img->line_length + x * (img->bits_per_pixel / 8));
	*(unsigned int *)dst = color;
}

int win_initialize(t_data *data, int width, int height)
{
	// minilibxの初期化
	data->mlx_ptr = mlx_init();
	if (!data->mlx_ptr) {
		perror("mlx_init");
		return (1);
	}

	// 幅 width, 高さ height のウィンドウを作成
	data->win_ptr = mlx_new_window(data->mlx_ptr, width, height, "10-second Window");
	if (!data->win_ptr) {
		perror("mlx_new_window");
		return (2);
	}
	
	data->img.img_ptr = mlx_new_image(data->mlx_ptr, width, height);
	if (!data->img.img_ptr) {
		perror("mlx_new_image");
		return (3);
	} else {
		data->img.adrs = mlx_get_data_addr(data->img.img_ptr, &data->img.bits_per_pixel, &data->img.line_length, &data->img.endian);
		data->img.w = width;
		data->img.h = height;
	}
	    
    return 0;
}

int render_next_frame(void* arg)
{
	t_data *data = (t_data*)arg;
	assert(data != NULL);

	void *img_ptr = data->img.img_ptr;
	assert(img_ptr != NULL);

	// 画面を転送
	mlx_put_image_to_window(data->mlx_ptr, data->win_ptr, img_ptr, 0, 0);
	// 画面をクリア
	my_img_clear(&data->img);
	// デバッグプリントをリセット
	debug_reset_lines();
	
	if (!(data->mnist.image_adrs)) return 0;

	unsigned char image_adrs[28 * 28];
	pthread_mutex_lock(&data->mutex);
	memcpy(image_adrs, data->mnist.image_adrs, sizeof(image_adrs));
	pthread_mutex_unlock(&data->mutex);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			unsigned char pixel = image_adrs[i * 28 + j];
			int gray = (int)pixel * 0x010101;
			my_img_pixel_put(&data->img, j + 100, i + 100, gray);
		}
	}

	pthread_mutex_lock(&data->mutex);
	size_t counter = data->mnist.counter;
	size_t epoch = data->mnist.epoch;
	float cost = data->mnist.cost;
	float accuracy = data->mnist.accuracy;
	pthread_mutex_unlock(&data->mutex);

	debug_print(data, "epoch=%zu, counter=%6zu, cost=%f, accuracy=%f", epoch, counter, cost, accuracy);

	return 0;
}

int	key_press_hook(int keycode, t_data *data)
{
	if (keycode == 65307) // 65307 is the keycode for ESC
	{
		// ループを正常に終了させる
		mlx_loop_end(data->mlx_ptr);
	}
	return (0);
}

void win_loop(t_data *data)
{
	mlx_loop_hook(data->mlx_ptr, render_next_frame, data);
	mlx_key_hook(data->win_ptr, key_press_hook, data);
	// mlx_loop_endが呼ばれると、このループは終了する
	mlx_loop(data->mlx_ptr);
	// --- クリーンアップ処理 ---
	mlx_destroy_image(data->mlx_ptr, data->img.img_ptr);
	mlx_destroy_window(data->mlx_ptr, data->win_ptr);
	mlx_destroy_display(data->mlx_ptr);
}
