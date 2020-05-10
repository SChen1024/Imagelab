#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"   // openv 头文件
#include <QDebug>
#include <QDir>
#include <QPainter>

void testYUVRotate();

cv::Mat gSrcImg;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 设定信号与槽 连接
    connect(ui->btn_test1,&QPushButton::clicked,this,&MainWindow::testFunc1);
    connect(ui->btn_test2,&QPushButton::clicked,this,&MainWindow::testFunc2);

    gSrcImg = cv::imread("./testimages/lena.png");

    // 初始化 ui
    ui->pt_log->clear();  // 清除框内输出


    // 测试部分函数
    // testYUVRotate();


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
using std::cout;

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

// 默认 尺寸为3的  均值滤波 // 自定义实现 暂时不考虑参数异常等 处理
cv::Mat meanFilter(const cv::Mat src, int ksize = 3)
{
    // 边界不处理, 直接忽略掉
    cv::Mat dst = src.clone();

    // 直接出, 强制向下取整, // 暴力计算每一个 邻域区间的值
    int k0 = ksize/2;
    int sum[3] = {0,0,0};
    for(int i=k0;i<dst.rows-k0-1;i++)
    {
        for(int j=k0;j<dst.cols-k0-1;j++)
        {
            // 清空 和数组
            memset(sum,0, sizeof(sum));

            // 计算三个通道的结果 和值 并计算 均值写入目标图像
            for(int c = 0;c<3;c++)
            {
                for(int m = 0;m<ksize;m++)
                {
                    for (int n=0;n<ksize;n++)
                    {
                        sum[c] += src.at<cv::Vec3b>(i-k0+m,j-k0+n)[c];
                    }
                }
                // 计算均值写入
                dst.at<cv::Vec3b>(i,j)[c] = cv::saturate_cast<uchar>((float)sum[c] /(ksize*ksize));
            }
        }
    }
    return dst;
}

// filter2D 实现 meanfilter
cv::Mat meanFilterByFilter2D(const cv::Mat src, int ksize = 3)
{
    cv::Mat kernel = (cv::Mat_<float>(ksize,ksize) << 1,1,1,1,1,1,1,1,1);
    kernel = kernel / 9.0f;
    cv::Mat dst;
    cv::filter2D(src,dst,src.depth(),kernel);
    return dst;
}

// 使用 blur 均值滤波
cv::Mat meanFilterByBlur(const cv::Mat src, int ksize = 3)
{
    cv::Mat dst;
    cv::blur(src,dst,cv::Size(ksize,ksize));

    return dst;
}

//中值滤波：C++ 代码实现 // 处理单通道图像 // 参考 https://www.cnblogs.com/ranjiewen/p/5699395.html
cv::Mat medianFilterGray(const cv::Mat &src, int ksize = 3)
{
    cv::Mat dst = src.clone();
    //0. 准备：获取图片的宽，高和像素信息，
    const int  num = ksize * ksize;
    std::vector<uchar> pixel(num);

    //相对于中心点，3*3领域中的点需要偏移的位置
    int delta[3 * 3][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 0 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, {1, 1}
    };
    //1. 中值滤波，没有考虑边缘
    for (int i = 1; i < src.rows - 1; ++i)
    {
        for (int j = 1; j < src.cols - 1; ++j)
        {
            //1.1 提取领域值 // 使用数组 这样处理 8邻域值 不适合更大窗口
            for (int k = 0; k < num; ++k)
            {
                pixel[k] = src.at<uchar>(i+delta[k][0], j+ delta[k][1]);
            }
            //1.2 排序  // 使用自带的库及排序即可
            std::sort(pixel.begin(), pixel.end());
            //1.3 获取该中心点的值
            dst.at<uchar>(i, j) = pixel[num / 2];
        }
    }
    return dst;
}

// 自定义两个像素的比较函数,  // 使用和值 排序
bool comp(const cv::Vec3b &p1, const cv::Vec3b &p2)
{
    return (p1[0] + p1[1] + p1[2]) < (p2[0] + p2[1] + p2[2]);
}
// 尝试彩色图像, 中值排序使用三个通道的和排序
cv::Mat medianFilterColor(const cv::Mat &src, int ksize = 3)
{
    cv::Mat dst = src.clone();
    //0. 准备：获取图片的宽，高和像素信息，
    const int  num = ksize * ksize;
    std::vector<cv::Vec3b> pixel(num);

    //相对于中心点，3*3领域中的点需要偏移的位置
    int delta[3 * 3][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 0 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, {1, 1}
    };
    //1. 中值滤波，没有考虑边缘
    for (int i = 1; i < src.rows - 1; ++i)
    {
        for (int j = 1; j < src.cols - 1; ++j)
        {
            //1.1 提取领域值 // 使用数组 这样处理 8邻域值 不适合更大窗口
            for (int k = 0; k < num; ++k)
            {
                pixel[k] = src.at<cv::Vec3b>(i + delta[k][0], j + delta[k][1]);
            }
            //1.2 排序  // 使用自定义的排序函数排序彩色图像
            std::sort(pixel.begin(),pixel.end(),comp);
            //1.3 获取该中心点的值
            dst.at<cv::Vec3b>(i, j) = pixel[num / 2];
        }
    }
    return dst;
}

// opencv 中值滤波
cv::Mat mediaFilterDefault(const cv::Mat &src, int ksize = 3)
{
    cv::Mat dst;
    cv::medianBlur(src, dst, ksize);
    return dst;
}


// 自适应中值滤波窗口实现  // 图像 计算座标, 窗口尺寸和 最大尺寸
uchar adaptiveProcess(const Mat &im, int row, int col, int kernelSize, int maxSize)
{
    std::vector<uchar> pixels;
    for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
        for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize); // 增大窗口尺寸，继续A过程。
        else
            return med;
    }
}
// 自适应均值滤波
cv::Mat adaptiveMediaFilter(const cv::Mat &src, int ksize = 3)
{
    int minSize = 3; // 滤波器窗口的起始尺寸
    int maxSize = 7; // 滤波器窗口的最大尺寸
    cv::Mat dst;
    // 扩展图像的边界
    cv::copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    // 图像循环
    for (int j = maxSize / 2; j < dst.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < dst.cols * dst.channels() - maxSize / 2; i++)
        {
            dst.at<uchar>(j, i) = adaptiveProcess(dst, j, i, minSize, maxSize);
        }
    }
    cv::Rect r = cv::Rect(cv::Point(maxSize / 2, maxSize / 2), cv::Point(dst.rows-maxSize / 2, dst.rows-maxSize / 2));
    cv::Mat res = dst(r);
    return res;
}

// 来源链接: https://www.cnblogs.com/wangguchangqing/p/6407717.html
void GaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma)
{
    CV_Assert(src.channels() || src.channels() == 3); // 只处理单通道或者三通道图像
    const static double pi = 3.1415926;
    // 根据窗口大小和sigma生成高斯滤波器模板
    // 申请一个二维数组，存放生成的高斯模板矩阵
    double **templateMatrix = new double*[ksize];
    for (int i = 0; i < ksize; i++)
        templateMatrix[i] = new double[ksize];
    int origin = ksize / 2; // 以模板的中心为原点
    double x2, y2;
    double sum = 0;
    for (int i = 0; i < ksize; i++)
    {
        x2 = pow(i - origin, 2);
        for (int j = 0; j < ksize; j++)
        {
            y2 = pow(j - origin, 2);
            // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            sum += g;
            templateMatrix[i][j] = g;
        }
    }
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            templateMatrix[i][j] /= sum;
            cout << templateMatrix[i][j] << " ";
        }
        cout << endl;
    }
    // 将模板应用到图像中
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int a = -border; a <= border; a++)
            {
                for (int b = -border; b <= border; b++)
                {
                    if (channels == 1)
                    {
                        sum[0] += templateMatrix[border + a][border + b] * dst.at<uchar>(i + a, j + b);
                    }
                    else if (channels == 3)
                    {
                        Vec3b rgb = dst.at<Vec3b>(i + a, j + b);
                        auto k = templateMatrix[border + a][border + b];
                        sum[0] += k * rgb[0];
                        sum[1] += k * rgb[1];
                        sum[2] += k * rgb[2];
                    }
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 释放模板数组
    for (int i = 0; i < ksize; i++)
        delete[] templateMatrix[i];
    delete[] templateMatrix;
}

// 来源链接: https://www.cnblogs.com/wangguchangqing/p/6407717.html
// 分离的 高斯滤波
void separateGaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma)
{
    CV_Assert(src.channels() == 1 || src.channels() == 3); // 只处理单通道或者三通道图像
    // 生成一维的高斯滤波模板
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for (int i = 0; i < ksize; i++)
    {
        // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
        double g = exp(-(i - origin) * (i - origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    // 归一化
    for (int i = 0; i < ksize; i++)
        matrix[i] /= sum;
    // 将模板应用到图像中
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    // 水平方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j + k); // 行不变，列变化；先做水平方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i, j + k);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 竖直方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i + k, j); // 列不变，行变化；竖直方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i + k, j);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete[] matrix;
}



// 将文件写入到 yuv 文件中 
bool WriteMat2YUV(const cv::Mat &img, std::string file_name, bool is_yuv_flg = false)
{
    int width = img.cols, height = img.rows;
    assert(width > 0 && height > 0);
    // 如果不是 YUV 格式的图 还需要转换成相应的格式 
    int data_len = width * height * 3 / 2;   // BGRBGR--> YUYV  w*h*3 --> w/2*h*4
    uchar *yuv_data = new uchar[data_len];

    // 将图像存储数组
    memcpy(yuv_data, img.data, data_len * sizeof(uchar));


    //  声明文件写入 指针 
    std::fstream pfile;
    pfile.open(file_name, std::ios::out | std::ios::binary);
    if (pfile)
    {
        pfile.write((const char*)yuv_data, data_len);
        pfile.close();
    }

    return true;
}



void testCV2Uchar()
{
    // 初始化全白图像, 然后绘制四条直线 彩色的 100*100*3
    const int TEST_SIZE = 100;
    cv::Mat test_image = cv::Mat(cv::Size(TEST_SIZE, TEST_SIZE), CV_8UC3,cv::Scalar(255,255,255));
    cv::line(test_image, cv::Point(0, 0), cv::Point(TEST_SIZE - 1, TEST_SIZE - 1), cv::Scalar(0, 255, 0));
    cv::line(test_image, cv::Point(0, TEST_SIZE - 1), cv::Point(TEST_SIZE - 1, 0), cv::Scalar(0, 0, 255));
    cv::line(test_image, cv::Point(TEST_SIZE/2, TEST_SIZE / 3), cv::Point(TEST_SIZE / 2, 2 * TEST_SIZE / 3), cv::Scalar(255, 0, 0));
    cv::line(test_image, cv::Point(TEST_SIZE / 3, TEST_SIZE / 2), cv::Point(2 * TEST_SIZE / 3, TEST_SIZE / 2), cv::Scalar(0, 255, 255));

    // 将 RGB 转换成 YUV2
    //uchar *data = new uchar[TEST_SIZE*TEST_SIZE];
    
    // 将 RGB 转换成 YUY2  200 * 100 * 1
    cv::Mat img_bgr, img_yuv;
    img_bgr = test_image.clone();
    cvtColor(img_bgr, img_yuv, cv::COLOR_RGB2YUV);

    const std::string TEST_IMG_DIR = "./testimages/yuv/";

    cv::imwrite(TEST_IMG_DIR + "test_yuv1.png", img_yuv);
    cv::imwrite(TEST_IMG_DIR + "test_bgr1.png", img_bgr);
    WriteMat2YUV(img_yuv, TEST_IMG_DIR + "test_yuv1.yuv");
    WriteMat2YUV(img_bgr, TEST_IMG_DIR + "test_bgr1.yuv");



}

void CvMat2UChar(const cv::Mat &src, uchar * image)
{

}

// 最后放 测试函数
// 测试函数
void testYUVRotate()
{
    testCV2Uchar();
}


/**
 * @fn  void rotateYUV(uchar* yuvdata,int width, int height, int cent_x,int cent_y,int r,int angle) QString compareImages(const cv::Mat &I1, const cv::Mat &I2, const QString str = "noise", const QString str_temp = "image-%1: psnr:%2, mssim: B:%3 G:%4 R:%5")
 *
 * @brief   Rotate yuv 
 *
 * @author  SChen
 * @date    2020-05-03
 *
 * @param [in,out]  yuvdata If non-null, the yuvdata
 * @param           width   The width
 * @param           height  The height
 * @param           cent_x  The cent x coordinate
 * @param           cent_y  The cent y coordinate
 * @param           r       An int to process
 * @param           angle   The angle
**/

void rotateYUV(uchar* yuvdata, int width, int height, int cent_x, int cent_y, int r, int angle)
{

}




// 对比两个图像 然后输出 参数信息
QString compareImages(const cv::Mat &I1,
    const cv::Mat &I2,
    const QString str = "noise",
    const QString str_temp = "image-%1: psnr:%2, mssim: B:%3 G:%4 R:%5")
{
    double psnr_ = getPSNR(I1, I2);
    cv::Scalar mssim_ = getMSSIM(I1, I2);

    // 根据 输出模板 生成参数信息
    QString res_str = str_temp.arg(str)
        .arg(psnr_)
        .arg(mssim_.val[0])
        .arg(mssim_.val[1])
        .arg(mssim_.val[2]);

    return res_str;
    // cv::imwrite(IMAGE_DIR + "dst_" + std::to_string(i + 1) + ".png", dst[i]);
}


// 全局 噪声图像数组, psnr 数组 mssim 数组
const std::string IMAGE_DIR ="./testimages/noise/";
std::vector<cv::Mat> gNoiseImg(6);
double psnr[6];
cv::Scalar mssim[6];
void MainWindow::testFunc1(void)
{
    // 用于读取 测试图片
    for(int i=0;i<6;i++)
    {
        gNoiseImg[i] = cv::imread(IMAGE_DIR + "lena-" + std::to_string(i+1) + ".png");
    }
    qDebug("ReadOK");
}

void MainWindow::testFunc2(void)
{
    // 测试 中值 滤波 三种方式的不同
    const int TEST = 1; // 使用统一的图进行测试 暂时使用 高 椒盐噪声图像
    QString res_str;

    // 噪声图像的参数值
    res_str = compareImages(gSrcImg, gNoiseImg[TEST]);
    ui->pt_log->appendPlainText(res_str);

    cv::Mat test_img = gNoiseImg[TEST];

    cv::Mat dst[4];

    // 测试 中值滤波 拆分三个通道进行中值滤波然后合并图像
    std::vector<cv::Mat> bgr(3);
    cv::split(test_img, bgr);
    bgr[0] = medianFilterGray(bgr[0]);
    bgr[1] = medianFilterGray(bgr[1]);
    bgr[2] = medianFilterGray(bgr[2]);

    cv::merge(bgr, dst[0]);     // 第一种方式
    dst[1] = medianFilterColor(test_img);   // 第二种 彩色直接 计算中值滤波
    dst[2] = mediaFilterDefault(test_img);  // opencv 实现 中值滤波

    // 拆分三个通道 计算自适应中值滤波
    cv::split(test_img, bgr);
    for (int i = 0; i < 3; i++)
        bgr[i] = adaptiveMediaFilter(bgr[i]);
    cv::merge(bgr, dst[3]);


    // 分别计算三种方式得到的滤波的效果 (结果图与 原始图比较)
    for(int i=0;i<4;i++)
    {
        res_str = compareImages(gSrcImg, dst[i]);
        // 噪声的参数值
        ui->pt_log->appendPlainText(res_str);

        cv::imwrite(IMAGE_DIR + "dst_media_" + std::to_string(i+1)+".png",dst[i]);
    }






//    // 测试 均值滤波 对于每一张图 都使用三种方式 进行滤波结果的展示
//    for(int i=0;i<6;i++)
//    {
//        psnr[i] = getPSNR(gSrcImg, gNoiseImg[i]);
//        mssim[i] = getMSSIM(gSrcImg, gNoiseImg[i]);
//        res_str = res_temp.arg(i+1)
//                            .arg(psnr[i])
//                            .arg(mssim[i].val[0])
//                            .arg(mssim[i].val[1])
//                            .arg(mssim[i].val[2]);
//        ui->pt_log->appendPlainText(res_str);






//        cv::imwrite("../testimages/noise/lena-" + std::to_string(i+1) + ".png", gNoiseImg[i]);
//    }




}


