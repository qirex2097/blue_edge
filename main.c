#include <mlx.h>

int main(void)
{
    void *mlx;
    void *win;

    // minilibxの初期化
    mlx = mlx_init();
    if (!mlx)
        return (1);

    // 幅800, 高さ600のウィンドウを作成
    win = mlx_new_window(mlx, 800, 600, "Empty Window");
    if (!win)
        return (1);

    // イベントループ開始
    mlx_loop(mlx);

    return (0);
}