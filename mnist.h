#ifndef _MNIST_H_
#define _MNIST_H_

unsigned char* mnist_read_images(const char* filename, int* count);
unsigned char* mnist_read_labels(const char* filename, int* count);
void mnist_print_image(unsigned char* images, unsigned char* labels, int index);

#endif // _MNIST_H_