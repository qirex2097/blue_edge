#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist.h"

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28

// 画像データを読み込む
t_mnist_data mnist_read_images(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return (t_mnist_data){0};
    }
    uint32_t magic, num, rows, cols;
    fread(&magic, 4, 1, fp);
    fread(&num, 4, 1, fp);
    fread(&rows, 4, 1, fp);
    fread(&cols, 4, 1, fp);
    // ビッグエンディアン→リトルエンディアン変換
    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);
    rows  = __builtin_bswap32(rows);
    cols  = __builtin_bswap32(cols);

    if (magic != 2051) {
        printf("MNIST image file\n");
        fclose(fp);
        return (t_mnist_data){0};
    }

    t_mnist_data data = { .adrs = NULL, .count = num, .cols = cols, .rows = rows, };
    data.adrs = malloc(num * rows * cols);
    fread(data.adrs, 1, num * rows * cols, fp);
    fclose(fp);
    return data;
}

// ラベルデータを読み込む
t_mnist_data mnist_read_labels(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return (t_mnist_data){0};
    }
    uint32_t magic, num;
    fread(&magic, 4, 1, fp);
    fread(&num, 4, 1, fp);
    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);
    if (magic != 2049) {
        printf("Not MNIST label file\n");
        fclose(fp);
        return (t_mnist_data){0};
    }

    t_mnist_data data = { .adrs = NULL, .count = num, .cols = 0, .rows = 0, };
    data.adrs = malloc(num);
    fread(data.adrs, 1, num, fp);
    fclose(fp);
    return data;
}

// 指定されたインデックスの画像とラベルを表示する
void mnist_print_image(unsigned char* images, unsigned char* labels, int index) {
    if (!images || !labels) return;

    printf("Image #%d, Label: %d\n", index, labels[index]);
    for (int i = 0; i < MNIST_IMAGE_ROWS; i++) {
        for (int j = 0; j < MNIST_IMAGE_COLS; j++) {
            unsigned char pixel = images[index * MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS + i * MNIST_IMAGE_COLS + j];
            printf("%c", pixel > 128 ? '#' : '.');
        }
        printf("\n");
    }
}

#ifdef MNIST_STANDALONE
int main(void)
{
    t_mnist_data images_data, labels_data;
    images_data = mnist_read_images("data/train-images-idx3-ubyte");
    labels_data = mnist_read_labels("data/train-labels-idx1-ubyte");

    if (images_data.adrs && labels_data.adrs) {
        printf("Loaded %d images, %d labels\n", images_data.count, labels_data.count);
        // 例: 1枚目の画像を表示
        mnist_print_image(images_data.adrs, labels_data.adrs, 0);
    }

    free(images_data.adrs);
    free(labels_data.adrs);
    return 0;
}
#endif // MNIST_STANDALONE
