#include "mainwindow.h"
#include <QApplication>

// 运行主窗口 用于显示界面 ui
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}


