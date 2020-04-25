#include "mainwindow.h"
#include <QApplication>
// 引入 opencv 函数头文件
#include <opencv2/opencv.hpp>

// 记录鼠标位置点, 以及 正在绘图标志位 flg 
cv::Point start_p(-1, -1), end_p(-1, -1);
bool flg_drawing = false;   

// 使用原始图像与临时图像 存储
cv::Mat src_img, temp_img;
 
//鼠标回调函数 // 记录窗口的x y 位置 
void on_MouseHandle(int event, int x, int y, int flags, void *param)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
    {
        start_p = cv::Point(x, y);      // 确定起始点
        temp_img = src_img.clone();     // 复制原始图, 进行绘图操作
        flg_drawing = true;
    }break;
    case cv::EVENT_MOUSEMOVE:
    {
        if (flg_drawing)
            end_p = cv::Point(x, y);        // 如果在绘制, 则更新移动后的目标点
    }break;
    case cv::EVENT_LBUTTONUP:
    {
        end_p = cv::Point(x, y);        // 确定最终点 
        src_img = temp_img.clone();     // 将图像更新成为原始图 存储下来
        flg_drawing = false;
    }break;
    }
}

// 返回两点之间的距离 直线距离 平方和的开方值
float distance(const cv::Point &p1, const cv::Point &p2)
{
    return cv::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    //MainWindow w;
    //w.show();
    // 设置 要显示的图像路径
    std::string img_lena = "./TestImages/lena.png";
    src_img = cv::imread(img_lena);

    std::string windows_name = "show";
    cv::namedWindow(windows_name,cv::WINDOW_AUTOSIZE);

    // 设置窗口 鼠标操作 监听 函数为 on_MouseHandle
    cv::setMouseCallback(windows_name, on_MouseHandle, 0);

    //初始化随机种子
    cv::RNG rng(time(0));

    while (true)
    {
        // 根据当前点 绘制
        if (flg_drawing)
        {
            temp_img = src_img.clone();

            cv::line(temp_img, start_p, end_p, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
            cv::rectangle(temp_img, cv::Rect(start_p, end_p), cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
            cv::circle(temp_img, start_p, distance(start_p,end_p), cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
            cv::imshow(windows_name, temp_img);
        }
        else
        {
            cv::imshow(windows_name, src_img);
        }


        // 设置 按 esc 退出循环
        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
    // return a.exec();
}






