#ifndef _MNIST_H_
#define _MNIST_H_

typedef struct {
    unsigned char* adrs;
    int count;
    int rows;
    int cols;
} t_mnist_data;

t_mnist_data mnist_read_images(const char* filename);
t_mnist_data mnist_read_labels(const char* filename);
void mnist_print_image(unsigned char* images, unsigned char* labels, int index);

#endif // _MNIST_H_