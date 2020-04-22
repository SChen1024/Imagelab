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
    std::string img_panda = "./TestImages/panda.png";
    std::string img_lena = "./TestImages/lena.png";

    // 读取两幅彩色图像  512*512
    cv::Mat panda_bgr = cv::imread(img_panda);
    cv::Mat lena_bgr = cv::imread(img_lena);
    // 声明结果图像 1020*1020
    cv::Mat res_bgr = cv::Mat::zeros(cv::Size(512,512), CV_8UC3);

    float a = 2.2f, b = 50;
    for (int i = 0; i < lena_bgr.rows; i++)
    {
        for (int j = 0; j < lena_bgr.cols; j++)
        {
            // 取出原始图像 灰度值
            cv::Vec3b tmp_px = lena_bgr.at<cv::Vec3b>(i, j);  
            // 每个通道减去最小值
            res_bgr.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(a * tmp_px[0] + b);
            res_bgr.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(a * tmp_px[1] + b);
            res_bgr.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(a * tmp_px[2] + b);
        }
    }
    cv::imshow("lena_bgr", lena_bgr);
    cv::imshow("res_bgr", res_bgr);

    cv::waitKey(0);

    return 0;
    // return a.exec();
}
