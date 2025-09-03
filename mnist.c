#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28

// 画像データを読み込む
unsigned char* mnist_read_images(const char* filename, int* count) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
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
        printf("Not MNIST image file\n");
        fclose(fp);
        return NULL;
    }
    *count = num;
    unsigned char* data = malloc(num * rows * cols);
    fread(data, 1, num * rows * cols, fp);
    fclose(fp);
    return data;
}

// ラベルデータを読み込む
unsigned char* mnist_read_labels(const char* filename, int* count) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }
    uint32_t magic, num;
    fread(&magic, 4, 1, fp);
    fread(&num, 4, 1, fp);
    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);
    if (magic != 2049) {
        printf("Not MNIST label file\n");
        fclose(fp);
        return NULL;
    }
    *count = num;
    unsigned char* labels = malloc(num);
    fread(labels, 1, num, fp);
    fclose(fp);
    return labels;
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
    int num_images, num_labels;
    unsigned char* images = mnist_read_images("data/train-images-idx3-ubyte", &num_images);
    unsigned char* labels = mnist_read_labels("data/train-labels-idx1-ubyte", &num_labels);

    if (images && labels) {
        printf("Loaded %d images, %d labels\n", num_images, num_labels);
        // 例: 1枚目の画像を表示
        mnist_print_image(images, labels, 0);
    }

    free(images);
    free(labels);
    return 0;
}
#endif // MNIST_STANDALONE
