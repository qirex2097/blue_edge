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
	size_t cols;
	size_t rows;
	unsigned char *image_adrs;
} t_mnist;

// mlxポインタをスレッドに渡すための構造体
typedef struct {
	void	*mlx_ptr;
	void	*win_ptr;
	t_img	img;
#if 0
	size_t	counter;
	size_t	cols;
	size_t	rows;
	unsigned char *image_adrs;
#else
	t_mnist mnist;
#endif
	pthread_mutex_t mutex;
} t_data;

#endif//MAIN_H_
