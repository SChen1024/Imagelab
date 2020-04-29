#include "mainwindow.h"
#include "ui_mainwindow.h"

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

void MainWindow::testFunc1(void)
{
    ui->pt_log->appendPlainText("你点击了 测试按钮 1 ");
}

void MainWindow::testFunc2(void)
{
    ui->pt_log->appendPlainText("你点击了 测试按钮 2");
}
