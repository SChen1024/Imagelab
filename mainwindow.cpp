#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include "opencv2/opencv.hpp"   // openv 头文件
#include <QDebug>
#include <QDir>
#include <QPainter>

cv::Mat gSrcImg;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 设定信号与槽 连接
    connect(ui->btn_test1,&QPushButton::clicked,this,&MainWindow::testFunc1);
    connect(ui->btn_test2,&QPushButton::clicked,this,&MainWindow::testFunc2);

    gSrcImg = cv::imread("../testimages/lena.png");

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

// 将图片缩放成 label 大小 然后显示
void ShowMatOnQtLabel(const cv::Mat & src_img, QLabel * label)
{
    cv::Mat tmp_resize_img;
    //  首先根据label 大小 缩放对应的图像 进行尺寸限制,此处可能 存在 缩放问题

    // 进行同比例缩放, 先满足 高度
    int height = label->height();
    int width = (int)((float)src_img.cols)/ ((float)src_img.rows)*height;


    cv::resize(src_img, tmp_resize_img, cv::Size(width, height));

    // 将原始图像转换成 QImage
    QImage img = CvMat2QImage(tmp_resize_img);

    // 将QImage 转换成 QPixmap
    QPixmap pixmap = QPixmap::fromImage(img);

    // 在相应的lable 上 显示图片
    label->setPixmap(pixmap);

}


#include <random>
using namespace  cv;

// 添加椒盐噪声 // 生成 随机 num 个 白点
void addSaltNoise(Mat &m, int num)
{
    // 随机数产生器
    std::random_device rd; //种子
    std::mt19937 gen(rd()); // 随机数引擎

    auto cols = m.cols * m.channels();

    for (int i = 0; i < num; i++)
    {
        auto row = static_cast<int>(gen() % m.rows);
        auto col = static_cast<int>(gen() % cols);

        auto p = m.ptr<uchar>(row);
        p[col++] = 255;
        p[col++] = 255;
        p[col] = 255;
    }
}

// 添加Gussia噪声
// 使用指针访问
void addGaussianNoise(Mat &m, int mu, int sigma)
{
    // 产生高斯分布随机数发生器
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> d(mu, sigma);

    auto rows = m.rows; // 行数
    auto cols = m.cols * m.channels(); // 列数

    for (int i = 0; i < rows; i++)
    {
        auto p = m.ptr<uchar>(i); // 取得行首指针
        for (int j = 0; j < cols; j++)
        {
            auto tmp = p[j] + d(gen);
            tmp = tmp > 255 ? 255 : tmp;
            tmp = tmp < 0 ? 0 : tmp;
            p[j] = tmp;
        }
    }
}

// 计算两幅图像的  PSNR
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);         // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}
// 计算两幅图像的 MSSIM
Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}



void MainWindow::testFunc1(void)
{
    // 添加椒盐噪声 并计算 PSNR和 SSIM
    cv::Mat salt_img;
    double psnr = 0;
    cv::Scalar mssim;

    QString res_temp = "Salt-%1 : psnr:%2, mssim: B:%3 G:%4 R:%5 ";
    QString res_str;

    // 计算三组图像的参数 0, 1000, 10000

    // 复制原始图像, 添加噪声, 计算 psnr和ssim  显示在 ui上
    salt_img = gSrcImg.clone();
    addSaltNoise(salt_img,0);

    psnr = getPSNR(gSrcImg, salt_img);
    mssim = getMSSIM(gSrcImg,salt_img);
    res_str = res_temp.arg(0)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    salt_img = gSrcImg.clone();
    addSaltNoise(salt_img,1000);

    psnr = getPSNR(gSrcImg, salt_img);
    mssim = getMSSIM(gSrcImg,salt_img);
    res_str = res_temp.arg(1000)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    // 左侧显示 1000 噪声 右侧显示 10000 噪声
    ShowMatOnQtLabel(salt_img,ui->lb_src);

    salt_img = gSrcImg.clone();
    addSaltNoise(salt_img,10000);

    psnr = getPSNR(gSrcImg, salt_img);
    mssim = getMSSIM(gSrcImg,salt_img);
    res_str = res_temp.arg(10000)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    ShowMatOnQtLabel(salt_img,ui->lb_dst);

}

void MainWindow::testFunc2(void)
{
    // 添加高斯噪声 并计算 PSNR和 SSIM
    cv::Mat guass_img;
    double psnr = 0;
    cv::Scalar mssim;

    QString res_temp = "gauss-%1- %2 : psnr:%3, mssim: B:%4 G:%5 R:%6 ";
    QString res_str;

    // 计算三组图像的参数 (0,1) (0,10), (10,1), (10,10)

    // 复制原始图像, 添加噪声, 计算 psnr和ssim  显示在 ui上
    guass_img = gSrcImg.clone();
    addGaussianNoise(guass_img,0,1);

    psnr = getPSNR(gSrcImg, guass_img);
    mssim = getMSSIM(gSrcImg,guass_img);
    res_str = res_temp.arg(0)
                        .arg(1)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    guass_img = gSrcImg.clone();
    addGaussianNoise(guass_img,0,10);

    psnr = getPSNR(gSrcImg, guass_img);
    mssim = getMSSIM(gSrcImg,guass_img);
    res_str = res_temp.arg(0)
                        .arg(10)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    guass_img = gSrcImg.clone();
    addGaussianNoise(guass_img,10,1);

    psnr = getPSNR(gSrcImg, guass_img);
    mssim = getMSSIM(gSrcImg,guass_img);
    res_str = res_temp.arg(10)
                        .arg(1)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);

    guass_img = gSrcImg.clone();
    addGaussianNoise(guass_img,10,10);

    psnr = getPSNR(gSrcImg, guass_img);
    mssim = getMSSIM(gSrcImg,guass_img);
    res_str = res_temp.arg(10)
                        .arg(10)
                        .arg(psnr)
                        .arg(mssim.val[0])
                        .arg(mssim.val[1])
                        .arg(mssim.val[2]);
    ui->pt_log->appendPlainText(res_str);



}


