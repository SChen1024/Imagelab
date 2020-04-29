#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include "opencv2/opencv.hpp"   // openv 头文件
#include <QDebug>
#include <QDir>
#include <QPainter>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 设定信号与槽 连接
    connect(ui->btn_test1,&QPushButton::clicked,this,&MainWindow::testFunc1);
    connect(ui->btn_test2,&QPushButton::clicked,this,&MainWindow::testFunc2);

    // 初始化 ui
    ui->pt_log->clear();  // 清除框内输出
}

MainWindow::~MainWindow()
{
    delete ui;
}

/**
 * @fn  QImage CvMat2QImage(const cv::Mat & mat)
 *
 * @brief   将opencv mat 转换成 QT image
 *
 * @author  IRIS_Chen
 * @date    2019/12/19
 *
 * @param   mat The matrix
 *
 * @return  A QImage
 */
QImage CvMat2QImage(const cv::Mat &mat)
{
    // 图像的通道
    int channel = mat.channels();

    // 设立一个表 直接查询 其中 0 2 是无效值 1 3 4 对应的转换值
    const std::map<int, QImage::Format> img_cvt_map {
        { 1, QImage::Format_Grayscale8 },
        { 3, QImage::Format_RGB888 },
        { 4, QImage::Format_ARGB32 }
    };

    QImage image(mat.data, mat.cols, mat.rows,
                 static_cast<int>(mat.step),
                 img_cvt_map.at(channel));

    // 三通道图像 值做 通道转换
    return channel == 3 ? image.rgbSwapped() : image;
}

/**
* @fn  static cv::Mat QImage2CvMat(const QImage &image);
*
* @brief   QT Image 转换成 cv Mat 结构
*
* @author  IRIS_Chen
* @date    2019/12/19
*
* @param   image   The image
*
* @return  A cv::Mat
*/
cv::Mat QImage2CvMat(const QImage &image)
{
    cv::Mat mat;
    const std::map<QImage::Format, int> img_cvt_map{
        { QImage::Format_Grayscale8, 1 },
        { QImage::Format_RGB888, 3 },
        { QImage::Format_ARGB32, 4}
    };

    return cv::Mat(image.height(), image.width(),img_cvt_map.at(image.format()));
}


// 图片路径
QString lena_img = "../testimages/lena.png";
void MainWindow::testFunc1(void)
{
    QPixmap pixmap;
    pixmap.load(lena_img);

    // 在图上绘制文字
    QPainter painter(&pixmap);
    painter.setPen(QColor(Qt::yellow));
    painter.drawText(100,100,"QT QPixmap");


    ui->lb_src->setPixmap(pixmap);
    ui->pt_log->appendPlainText("左侧使用 QPixmap load 图像数据1 ");
}

void MainWindow::testFunc2(void)
{
    cv::Mat mat = cv::imread("../testimages/lena.png");
    // 在图上显示文字
    cv::putText(mat,"OpenCV Mat",cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX,1.0, cv::Scalar(0, 255, 255));

    QImage image = CvMat2QImage(mat);

    ui->lb_dst->setPixmap(QPixmap::fromImage(image));
    ui->pt_log->appendPlainText("右侧使用 Mat --> QImage --> QPixmap 进行显示2 ");
}
