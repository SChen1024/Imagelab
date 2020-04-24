#include "mainwindow.h"
#include <QApplication>
// 引入 opencv 函数头文件
#include <opencv2/opencv.hpp>
int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    //MainWindow w;
    //w.show();
    // 设置 要显示的图像路径
    std::string img_lena = "./TestImages/lena.png";

    // 读取两幅彩色图像  512*512
    cv::Mat lena_bgr = cv::imread(img_lena);
    // 声明结果图像 1020*1020
    cv::Mat res_bgr = cv::Mat::zeros(cv::Size(512,512), CV_8UC3);

    // 绘制基本图形
    cv::line(lena_bgr, cv::Point(100, 200), cv::Point(500, 300), cv::Scalar(0, 255, 0));

    cv::line(lena_bgr, cv::Point(100, 200), cv::Point(500, 300), cv::Scalar(0, 255, 0), 1, 8, 1);
    cv::line(lena_bgr, cv::Point(100, 200), cv::Point(500, 300), cv::Scalar(0, 255, 0), 1, 8, 2);
    cv::line(lena_bgr, cv::Point(100, 200), cv::Point(500, 300), cv::Scalar(0, 255, 0), 1, 8, 3);


    cv::waitKey(0);

    return 0;
    // return a.exec();
}
