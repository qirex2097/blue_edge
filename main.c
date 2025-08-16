#include <stdio.h>
#include "main.h"
#include "gate.h"
#include "win.h"

int	main(void)
{
	t_data      data;
	int result;
	
	result = win_initialize(&data, 800, 600);
	if (result != 0)
		return result;
	
	result = gate_initialize(&data);
	if (result != 0)
		return result;
	
	// イベントループ開始
	win_loop(&data);

	printf("Program finished gracefully.\n");
	
	return (0);
}
