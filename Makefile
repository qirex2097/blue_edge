NAME = app
SRC = main.c gate.c win.c
CC = gcc
MLX_DIR = ./minilibx-linux
CFLAGS = -Wall -Wextra -Werror -I$(MLX_DIR)
LDFLAGS = -L$(MLX_DIR) -lmlx -lXext -lX11 -lm -lpthread

all: $(NAME)

$(NAME): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(NAME)

clean:
	rm -f $(NAME)

re: clean all

.PHONY: all clean re
