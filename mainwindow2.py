# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI2.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1680, 841)
        MainWindow.setMouseTracking(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMouseTracking(True)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setAutoFillBackground(True)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1654, 710))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.scrollAreaWidgetContents)
        self.scrollArea_2.setGeometry(QtCore.QRect(0, 0, 1871, 791))
        self.scrollArea_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.scrollArea_2.setAutoFillBackground(True)
        self.scrollArea_2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 1869, 789))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.slider1 = QtWidgets.QScrollBar(self.scrollAreaWidgetContents_2)
        self.slider1.setGeometry(QtCore.QRect(30, 230, 551, 20))
        self.slider1.setOrientation(QtCore.Qt.Horizontal)
        self.slider1.setObjectName("slider1")
        self.channel1 = PlotWidget(self.scrollAreaWidgetContents_2)
        self.channel1.setGeometry(QtCore.QRect(30, 20, 551, 211))
        self.channel1.setObjectName("channel1")
        self.channel2 = PlotWidget(self.scrollAreaWidgetContents_2)
        self.channel2.setGeometry(QtCore.QRect(30, 480, 551, 211))
        self.channel2.setObjectName("channel2")
        self.channel5 = PlotWidget(self.scrollAreaWidgetContents_2)
        self.channel5.setGeometry(QtCore.QRect(1180, 170, 471, 351))
        self.channel5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.channel5.setAutoFillBackground(True)
        self.channel5.setObjectName("channel5")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 270, 1121, 191))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalSlider_10 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_10.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_10.setObjectName("verticalSlider_10")
        self.horizontalLayout_2.addWidget(self.verticalSlider_10)
        self.verticalSlider_9 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_9.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_9.setObjectName("verticalSlider_9")
        self.horizontalLayout_2.addWidget(self.verticalSlider_9)
        self.verticalSlider_8 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_8.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_8.setObjectName("verticalSlider_8")
        self.horizontalLayout_2.addWidget(self.verticalSlider_8)
        self.verticalSlider_7 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_7.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_7.setObjectName("verticalSlider_7")
        self.horizontalLayout_2.addWidget(self.verticalSlider_7)
        self.verticalSlider_6 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_6.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_6.setObjectName("verticalSlider_6")
        self.horizontalLayout_2.addWidget(self.verticalSlider_6)
        self.verticalSlider_5 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_5.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_5.setObjectName("verticalSlider_5")
        self.horizontalLayout_2.addWidget(self.verticalSlider_5)
        self.verticalSlider_4 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_4.setObjectName("verticalSlider_4")
        self.horizontalLayout_2.addWidget(self.verticalSlider_4)
        self.verticalSlider_3 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_3")
        self.horizontalLayout_2.addWidget(self.verticalSlider_3)
        self.verticalSlider_2 = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.horizontalLayout_2.addWidget(self.verticalSlider_2)
        self.verticalSlider = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.horizontalLayout_2.addWidget(self.verticalSlider)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(1180, 620, 471, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.comboBox)
        self.checkBox = QtWidgets.QCheckBox(self.scrollAreaWidgetContents_2)
        self.checkBox.setGeometry(QtCore.QRect(1180, 130, 131, 21))
        self.checkBox.setObjectName("checkBox")
        self.channel3 = PlotWidget(self.scrollAreaWidgetContents_2)
        self.channel3.setGeometry(QtCore.QRect(600, 20, 551, 211))
        self.channel3.setObjectName("channel3")
        self.channel4 = PlotWidget(self.scrollAreaWidgetContents_2)
        self.channel4.setGeometry(QtCore.QRect(600, 480, 551, 211))
        self.channel4.setObjectName("channel4")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(1180, 520, 471, 41))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAutoFillBackground(True)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.horizontalSlider = QtWidgets.QSlider(self.horizontalLayoutWidget_3)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_4.addWidget(self.horizontalSlider)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(1180, 570, 471, 41))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.horizontalLayoutWidget_4)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalLayout_5.addWidget(self.horizontalSlider_2)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1680, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuLabel = QtWidgets.QMenu(self.menubar)
        self.menuLabel.setObjectName("menuLabel")
        self.menuWaveform = QtWidgets.QMenu(self.menubar)
        self.menuWaveform.setObjectName("menuWaveform")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuDevice = QtWidgets.QMenu(self.menubar)
        self.menuDevice.setObjectName("menuDevice")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuAudio = QtWidgets.QMenu(self.menubar)
        self.menuAudio.setObjectName("menuAudio")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.slowdown = QtWidgets.QToolBar(MainWindow)
        self.slowdown.setObjectName("slowdown")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.slowdown)
        self.toolBar_3 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_3.setObjectName("toolBar_3")
        MainWindow.addToolBar(QtCore.Qt.BottomToolBarArea, self.toolBar_3)
        self.actionChannel_2 = QtWidgets.QAction(MainWindow)
        self.actionChannel_2.setObjectName("actionChannel_2")
        self.actionChannel_3 = QtWidgets.QAction(MainWindow)
        self.actionChannel_3.setObjectName("actionChannel_3")
        self.play = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.play.setIcon(icon)
        self.play.setObjectName("play")
        self.spectrogram = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/spec.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.spectrogram.setIcon(icon1)
        self.spectrogram.setObjectName("spectrogram")
        self.zoomin = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icons/zoom-in-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoomin.setIcon(icon2)
        self.zoomin.setObjectName("zoomin")
        self.zoomout = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icons/zoom-out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoomout.setIcon(icon3)
        self.zoomout.setObjectName("zoomout")
        self.pause = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icons/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pause.setIcon(icon4)
        self.pause.setObjectName("pause")
        self.actionZoom_out_2 = QtWidgets.QAction(MainWindow)
        self.actionZoom_out_2.setObjectName("actionZoom_out_2")
        self.actionChannel_1 = QtWidgets.QAction(MainWindow)
        self.actionChannel_1.setObjectName("actionChannel_1")
        self.generateReport = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icons/printer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.generateReport.setIcon(icon5)
        self.generateReport.setObjectName("generateReport")
        self.delete_2 = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icons/remove.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.delete_2.setIcon(icon6)
        self.delete_2.setObjectName("delete_2")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.actionSpeed_Up = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icons/arrow-up.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSpeed_Up.setIcon(icon7)
        self.actionSpeed_Up.setObjectName("actionSpeed_Up")
        self.actionSpeed_Down = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icons/down-left-arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSpeed_Down.setIcon(icon8)
        self.actionSpeed_Down.setObjectName("actionSpeed_Down")
        self.actionNew_Window = QtWidgets.QAction(MainWindow)
        self.actionNew_Window.setObjectName("actionNew_Window")
        self.actionPlay_Audio = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icons/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPlay_Audio.setIcon(icon9)
        self.actionPlay_Audio.setObjectName("actionPlay_Audio")
        self.actionPause_Audio = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("icons/stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPause_Audio.setIcon(icon10)
        self.actionPause_Audio.setObjectName("actionPause_Audio")
        self.menuFile.addAction(self.actionNew_Window)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionChannel_1)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.generateReport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.exit)
        self.menuWaveform.addAction(self.play)
        self.menuWaveform.addAction(self.pause)
        self.menuWaveform.addAction(self.delete_2)
        self.menuWaveform.addSeparator()
        self.menuWaveform.addAction(self.zoomin)
        self.menuWaveform.addAction(self.zoomout)
        self.menuWaveform.addSeparator()
        self.menuWaveform.addAction(self.actionSpeed_Up)
        self.menuWaveform.addAction(self.actionSpeed_Down)
        self.menuWaveform.addSeparator()
        self.menuView.addAction(self.spectrogram)
        self.menuAudio.addAction(self.actionPlay_Audio)
        self.menuAudio.addAction(self.actionPause_Audio)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuLabel.menuAction())
        self.menubar.addAction(self.menuWaveform.menuAction())
        self.menubar.addAction(self.menuAudio.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuDevice.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.slowdown.addAction(self.generateReport)
        self.slowdown.addSeparator()
        self.slowdown.addAction(self.play)
        self.slowdown.addAction(self.pause)
        self.slowdown.addAction(self.delete_2)
        self.slowdown.addSeparator()
        self.slowdown.addAction(self.zoomin)
        self.slowdown.addAction(self.zoomout)
        self.slowdown.addSeparator()
        self.slowdown.addAction(self.actionSpeed_Up)
        self.slowdown.addAction(self.actionSpeed_Down)
        self.slowdown.addSeparator()
        self.slowdown.addAction(self.spectrogram)
        self.slowdown.addSeparator()
        self.slowdown.addAction(self.actionPlay_Audio)
        self.slowdown.addAction(self.actionPause_Audio)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Color Map"))
        self.comboBox.setItemText(0, _translate("MainWindow", "plasma"))
        self.comboBox.setItemText(1, _translate("MainWindow", "inferno"))
        self.comboBox.setItemText(2, _translate("MainWindow", "magma"))
        self.comboBox.setItemText(3, _translate("MainWindow", "bone"))
        self.comboBox.setItemText(4, _translate("MainWindow", "gray"))
        self.checkBox.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.label.setWhatsThis(_translate("MainWindow", "set"))
        self.label.setText(_translate("MainWindow", "Set Min"))
        self.label_2.setWhatsThis(_translate("MainWindow", "set"))
        self.label_2.setText(_translate("MainWindow", "Set Max"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuLabel.setTitle(_translate("MainWindow", "Label"))
        self.menuWaveform.setTitle(_translate("MainWindow", "Waveform"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuDevice.setTitle(_translate("MainWindow", "Device"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuAudio.setTitle(_translate("MainWindow", "Audio"))
        self.slowdown.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.slowdown.setToolTip(_translate("MainWindow", "Pause"))
        self.toolBar_3.setWindowTitle(_translate("MainWindow", "toolBar_3"))
        self.actionChannel_2.setText(_translate("MainWindow", "Add Channel 2"))
        self.actionChannel_2.setShortcut(_translate("MainWindow", "Ctrl+2"))
        self.actionChannel_3.setText(_translate("MainWindow", "Add Channel 3"))
        self.actionChannel_3.setShortcut(_translate("MainWindow", "Ctrl+3"))
        self.play.setText(_translate("MainWindow", "Play"))
        self.play.setToolTip(_translate("MainWindow", "Play"))
        self.play.setStatusTip(_translate("MainWindow", "play"))
        self.play.setWhatsThis(_translate("MainWindow", "play"))
        self.play.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.spectrogram.setText(_translate("MainWindow", "Spectrogram"))
        self.spectrogram.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.zoomin.setText(_translate("MainWindow", "Zoom in"))
        self.zoomin.setShortcut(_translate("MainWindow", "Up"))
        self.zoomout.setText(_translate("MainWindow", "Zoom out"))
        self.zoomout.setShortcut(_translate("MainWindow", "Down"))
        self.pause.setText(_translate("MainWindow", "Pause"))
        self.pause.setToolTip(_translate("MainWindow", "Pause"))
        self.pause.setStatusTip(_translate("MainWindow", "Pause"))
        self.pause.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionZoom_out_2.setText(_translate("MainWindow", "Zoom out"))
        self.actionChannel_1.setText(_translate("MainWindow", "Add Audio File"))
        self.actionChannel_1.setShortcut(_translate("MainWindow", "Ctrl+1"))
        self.generateReport.setText(_translate("MainWindow", "Generate Report"))
        self.generateReport.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.delete_2.setText(_translate("MainWindow", "Remove"))
        self.delete_2.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.exit.setText(_translate("MainWindow", "Exit"))
        self.exit.setShortcut(_translate("MainWindow", "Esc"))
        self.actionSpeed_Up.setText(_translate("MainWindow", "Speed Up"))
        self.actionSpeed_Up.setShortcut(_translate("MainWindow", "Ctrl+U"))
        self.actionSpeed_Down.setText(_translate("MainWindow", "Slow Down"))
        self.actionSpeed_Down.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actionNew_Window.setText(_translate("MainWindow", "New Window"))
        self.actionPlay_Audio.setText(_translate("MainWindow", "Play Audio"))
        self.actionPlay_Audio.setShortcut(_translate("MainWindow", "Ctrl+Shift+P"))
        self.actionPause_Audio.setText(_translate("MainWindow", "Pause Audio"))
        self.actionPause_Audio.setShortcut(_translate("MainWindow", "Ctrl+Shift+O"))
from pyqtgraph import *
from PyQt5.QtWidgets import QFileDialog
