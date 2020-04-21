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
    std::string test_pic = "./TestImages/lena.png";

    // 读取图像
    cv::Mat lena_rgb = cv::imread(test_pic);

    //  生命三个灰色的图像
    cv::Mat lena_gray_avg = cv::Mat::zeros(lena_rgb.size(), CV_8UC1);
    cv::Mat lena_gray_weighted = cv::Mat::zeros(lena_rgb.size(), CV_8UC1);
    cv::Mat lena_gray_shift = cv::Mat::zeros(lena_rgb.size(), CV_8UC1);

    // 遍历每一个像素进行灰度化
    for (int i = 0; i < lena_rgb.rows; i++)
    {
        for (int j = 0; j < lena_rgb.cols; j++)
        {
            cv::Vec3b tmp_px = lena_rgb.at<cv::Vec3b>(i, j);
            lena_gray_avg.at<uchar>(i, j) = (uchar)((tmp_px[0] + tmp_px[1] + tmp_px[2]) / 3);
            lena_gray_weighted.at<uchar>(i, j) = (uchar)((0.299f * tmp_px[0] + 0.587f * tmp_px[1] + 0.114f * tmp_px[2]));
            lena_gray_shift.at<uchar>(i, j) = (uchar)((38 * tmp_px[0] + 75 * tmp_px[1] + 15 * tmp_px[2]) >> 7);
        }
    }

    // 显示图像
    cv::imshow("lena_rgb", lena_rgb);
    cv::imshow("lena_gray_avg", lena_gray_avg);
    cv::imshow("lena_gray_weighted", lena_gray_weighted);
    cv::imshow("lena_gray_shift", lena_gray_shift);

    cv::waitKey(0);


    return a.exec();
}
