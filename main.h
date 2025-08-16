#ifndef MAIN_H_
#define MAIN_H_
#include <stdlib.h>
// mlxポインタをスレッドに渡すための構造体
typedef struct s_data
{
	void	*mlx;
	void	*win;
	size_t	counter;
	size_t  prev_counter;
}           t_data;

#endif//MAIN_H_
