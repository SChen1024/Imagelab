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
    //std::string test_pic = "./TestImages/lena.png";
    double time_cnt = 0;
    double time_s = 0.0;

    // 读取图像
    // cv::Mat lena_rgb = cv::imread(test_pic);
    // 声明 彩色图像 和灰度图像  // 设置 10000*10000 尺寸的图像, 避免出错
    cv::Mat img_bgr = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);

    time_cnt = cv::getTickCount();
    // 遍历每一个像素进行灰度化
    for (int i = 0; i < img_bgr.rows; i++)
    {
        for (int j = 0; j < img_bgr.cols; j++)
        {
            img_bgr.at<cv::Vec3b>(i, j)[0] = 0;
        }
    }
    time_s = ((double)cv::getTickCount() - time_cnt) / cv::getTickFrequency();
    printf("index scan image time: \t\t %f second \n", time_s);

    time_cnt = cv::getTickCount();
    // 使用指针进行图像访问
    for (int i = 0; i < img_bgr.rows; i++)
    {
        cv::Vec3b *p_bgr = img_bgr.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img_bgr.cols; j++)
        {
            p_bgr[j][0] = 0;    // 访问(i,j) 的第一个通道
        }
    }
    time_s = ((double)cv::getTickCount() - time_cnt) / cv::getTickFrequency();
    printf("pointer scan image time: \t %f second \n", time_s);

    time_cnt = cv::getTickCount();
    // 使用迭代器访问
    for (cv::Mat_<cv::Vec3b>::iterator it = img_bgr.begin<cv::Vec3b>();
        it != img_bgr.end<cv::Vec3b>(); it++)
    {
        (*it)[0] = 0;
    }
    time_s = ((double)cv::getTickCount() - time_cnt) / cv::getTickFrequency();
    printf("iterator scan image time: \t %f second \n", time_s);

    cv::waitKey(0);

    return 0;
    // return a.exec();
}
