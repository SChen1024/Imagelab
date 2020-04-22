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
    // 声明结果图像
    cv::Mat res_bgr = cv::Mat::zeros(lena_bgr.size(), CV_8UC3);

    for (int i = 0; i < lena_bgr.rows; i++)
    {
        for (int j = 0; j < lena_bgr.cols; j++)
        {
            res_bgr.at<cv::Vec3b>(i, j)[0] = (panda_bgr.at<cv::Vec3b>(i, j)[0] + lena_bgr.at<cv::Vec3b>(i, j)[0]) / 2;
            res_bgr.at<cv::Vec3b>(i, j)[1] = (panda_bgr.at<cv::Vec3b>(i, j)[1] + lena_bgr.at<cv::Vec3b>(i, j)[1]) / 2;
            res_bgr.at<cv::Vec3b>(i, j)[2] = (panda_bgr.at<cv::Vec3b>(i, j)[2] + lena_bgr.at<cv::Vec3b>(i, j)[2]) / 2;
        }
    }

    cv::imshow("panda_bgr", panda_bgr);
    cv::imshow("lena_bgr", lena_bgr);
    cv::imshow("res_bgr", res_bgr);

    cv::waitKey(0);

    return 0;
    // return a.exec();
}
