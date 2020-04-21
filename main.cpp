#include "mainwindow.h"

#include <QApplication>


// 引入 opencv 函数头文件
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();


    // 设置 要显示的图像路径
    std::string test_pic = "../TestImages/lena.png";

    // 读取图像
    cv::Mat lena_img = cv::imread(test_pic);

    // 显示图像
    cv::imshow("图像显示窗口", lena_img);
    // cv::waitKey(100);

    // 存储图像
    std::string write_pic = "../TestImages/lenalena_write.png";

    cv::imwrite(write_pic,lena_img);

    return a.exec();
}
