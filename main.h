#ifndef MAIN_H_
#define MAIN_H_
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    void *img_ptr;
    char *adrs;
    int bits_per_pixel;
    int line_length;
    int endian;
    int w, h;
} t_img;

typedef struct {
	size_t counter;
	unsigned char image_adrs[28 * 28];
} t_mnist;

// mlxポインタをスレッドに渡すための構造体
typedef struct {
	void	*mlx_ptr;
	void	*win_ptr;
	t_img	img;
	t_mnist mnist;
	pthread_mutex_t mutex;
} t_data;

#endif//MAIN_H_
