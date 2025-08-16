#ifndef WIN_H_
#define WIN_H_
#include "main.h"

void debug_reset_lines(void);
void debug_print(t_data *data, const char *line, ...);

int win_initialize(t_data *data, int width, int height);
void win_loop(t_data *data);

#endif//WIN_H_
