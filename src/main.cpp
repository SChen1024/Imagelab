#include "mainwindow.h"
#include <QApplication>
// 引入 opencv 函数头文件
#include <opencv2/opencv.hpp>

// 进行 测试 算法
cv::Mat testFunc(const cv::Mat &src_img)
{
    cv::Mat res_img = cv::Mat::zeros(src_img.size(), CV_8UC1);

    for (int i = 1; i < src_img.rows - 1; i++)
    {
        for (int j = 1; j < src_img.cols - 1; j++)
        {
            res_img.at<uchar>(i, j) = cv::saturate_cast<uchar>(src_img.at<uchar>(i, j)
                + src_img.at<uchar>(i, j) - src_img.at<uchar>(i - 1, j)
                + src_img.at<uchar>(i, j) - src_img.at<uchar>(i + 1, j)
                + src_img.at<uchar>(i, j) - src_img.at<uchar>(i, j - 1)
                + src_img.at<uchar>(i, j) - src_img.at<uchar>(i, j + 1));
        }
    }
    return res_img;
}
// 使用测试 指针函数
cv::Mat testFunc2(const cv::Mat &src_img)
{
    cv::Mat res_img = cv::Mat::zeros(src_img.size(), CV_8UC1);

    for (int i = 1; i < src_img.rows - 1; i++)
    {
        const uchar* p_row_pre = src_img.ptr<uchar>(i - 1);
        const uchar* p_row_cur = src_img.ptr<uchar>(i);
        const uchar* p_row_next = src_img.ptr<uchar>(i + 1);

        uchar* p_row_res = res_img.ptr<uchar>(i);
        for (int j = 1; j < src_img.cols - 1; j++)
        {
            *p_row_res++ = cv::saturate_cast<uchar>(5 * p_row_cur[j]
                - p_row_cur[j-1] - p_row_cur[j+1] - p_row_pre[j] - p_row_next[j]);
        }
    }
    return res_img;
}

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

    // 设置 要显示的图像路径
    std::string lena_png = "../TestImages/lena.png";
    cv::Mat src_img = cv::imread(lena_png);
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);

    // 测试索引方式进行 锐化运算
    double t = (double)cv::getTickCount();
    cv::Mat res_img = testFunc(src_img);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sharpen-index: \t\t" << t << std::endl;

    // 测试 指针方式进行 锐化运算
    t = (double)cv::getTickCount();
    res_img = testFunc2(src_img);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sharpen-pointer: \t" << t << std::endl;
    
    cv::Mat kernel = (cv::Mat_<char>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::Mat res_img2;

    // 测试 filter 2D 算法时间
    t = (double)cv::getTickCount();
    cv::filter2D(src_img, res_img2, src_img.depth(), kernel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sharpen-filter: \t" << t << std::endl;

    cv::imshow("src_img", src_img);
    cv::imshow("res_img", res_img);
    cv::imshow("res_img2", res_img2);
    cv::waitKey(0);
    return 0;
    // return a.exec();
}


