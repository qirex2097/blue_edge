NAME = app
SRCS = main.c gate.c win.c mnist.c
OBJS = $(SRCS:.c=.o)
CC = gcc
MLX_DIR = ./minilibx-linux
CFLAGS = -Wall -Wextra -Werror -I$(MLX_DIR)
LDFLAGS = -L$(MLX_DIR) -lmlx -lXext -lX11 -lm -lpthread

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $(NAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

mnist: mnist.c
	$(CC) mnist.c -D MNIST_STANDALONE

clean:
	rm -f $(OBJS)
	
fclean:
	rm -f $(NAME) a.out

re: fclean all

.PHONY: all clean re
