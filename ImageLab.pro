QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

#  显示控控制台窗口, 用于调试
# CONFIG += console

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# 设置 ui 路径
# UI_DIR += $$PWD/src/ui/

#  引入各种库 win32
win32{

    # 独立依赖文件夹,
    DEPENDS     = $$PWD/depend/

    # 动态添加， 不需要本地配置环境
    OPENCV_LIB  = -L$$DEPENDS/opencv/lib/ -lopencv_world430
    OPENCV_LIBd = -L$$DEPENDS/opencv/lib/ -lopencv_world430d
    OPENCV_INC  =  $$DEPENDS/opencv/include

    # 添加库
    win32:CONFIG(release, debug|release): LIBS += $$OPENCV_LIB
    else:win32:CONFIG(debug, debug|release): LIBS += $$OPENCV_LIBd
}




# 加入依赖库和依赖头文件
# LIBS += $$OPENCV_LIB
INCLUDEPATH += $$OPENCV_INC
DEPENDPATH += $$OPENCV_INC



# 主要的文件
FORMS += \
    mainwindow.ui

HEADERS += \
    mainwindow.h

SOURCES += \
    main.cpp \
    mainwindow.cpp

