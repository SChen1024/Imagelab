#include <iostream>

// 引入 opencv 函数头文件
#include <opencv2/opencv.hpp>

int main()
{
    // 设置 要显示的图像路径
    std::string test_pic = "D:\\Project\\Vision\\ImageLab\\TestImages\\lena.png";

    // 读取图像
    cv::Mat lena_img = cv::imread(test_pic);

    // 显示图像
    cv::imshow("图像显示窗口", lena_img);
    cv::waitKey(0);

    return 0;
}

