# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cam.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets, uic
from mplwidget import MplWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # Importacion de la GUI nueva
        uic.loadUi('GUI_New.ui', self)

        MainWindow.setObjectName("MainWindow")
        font = QtGui.QFont()
        font.setPointSize(14)
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget.setCurrentIndex(0)

        self.btn_start = QtWidgets.QPushButton(self.tab)
        self.btn_start.setGeometry(QtCore.QRect(10, 10, 100, 100))

        self.btn_restart = QtWidgets.QPushButton(self.tab)
        self.btn_restart.setGeometry(QtCore.QRect(115, 10, 100, 100))

        self.btn_photo = QtWidgets.QPushButton(self.tab)
        self.btn_photo.setGeometry(QtCore.QRect(220, 10, 100, 100))

        self.btn_quit = QtWidgets.QPushButton(self.tab)
        self.btn_quit.setGeometry(QtCore.QRect(325, 10, 100, 100))

        # Cajita de numero de frames
        self.spinBox = QtWidgets.QSpinBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.spinBox.sizePolicy().hasHeightForWidth())
        self.spinBox.setSizePolicy(sizePolicy)
        self.spinBox.setMinimumSize(QtCore.QSize(0, 50))
        self.spinBox.setRange(1, 1800)
        self.spinBox.setProperty("value", 15)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setGeometry(QtCore.QRect(10, 120, 100, 50))

        # Checkbox de captura automatica
        self.checkBox = QtWidgets.QCheckBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox.setChecked(False)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setGeometry(QtCore.QRect(115, 120, 100, 50))

        # Cuadro de la imagen RGB
        self.label_rgb = QtWidgets.QLabel(self.tab)
        self.label_rgb.setGeometry(QtCore.QRect(10, 210, 401, 441))
        self.label_rgb.setText("")
        self.label_rgb.setObjectName("label_rgb")

        # Cuadro de la imagen Depth
        self.label_depth = QtWidgets.QLabel(self.tab)
        self.label_depth.setGeometry(QtCore.QRect(420, 210, 401, 441))
        self.label_depth.setText("")
        self.label_depth.setObjectName("label_depth")

        # Widget de matplotlib grafico de clasificacion tiempo real
        self.MplWidget = MplWidget(self.tab)
        self.MplWidget.setGeometry(QtCore.QRect(822, 220, 500, 441))
        self.MplWidget.setObjectName("MplWidget")

        self.MplWidget1 = MplWidget(self.tab_2)
        self.MplWidget1.setObjectName("MplWidget1")
        self.MplWidget1.setGeometry(QtCore.QRect(10, 10, 665, 620))

        self.MplWidget2 = MplWidget(self.tab_2)
        self.MplWidget2.setObjectName("MplWidget2")
        self.MplWidget2.setGeometry(QtCore.QRect(685, 10, 665, 620))

        self.MplWidget3 = MplWidget(self.tab_3)
        self.MplWidget3.setObjectName("MplWidget3")
        self.MplWidget3.setGeometry(QtCore.QRect(10, 10, 665, 620))

        self.MplWidget4 = MplWidget(self.tab_3)
        self.MplWidget4.setObjectName("MplWidget4")
        self.MplWidget4.setGeometry(QtCore.QRect(685, 10, 665, 620))


        self.label_den = QtWidgets.QLabel(self.tab)
        self.label_den.setGeometry(QtCore.QRect(832, 95, 120, 20))
        self.label_den.setText("Density:")

        # Text box
        self.density = QtWidgets.QLineEdit(self.tab)
        self.density.setGeometry(QtCore.QRect(832, 120, 120, 50))
        self.density.setText("2.50")
        self.density.setObjectName("density")

        # Label Volumen
        self.volume = QtWidgets.QLabel(self.tab)
        self.volume.setGeometry(QtCore.QRect(832, 180, 400, 60))
        self.volume.setText("VOLUME:  U_v")
        self.volume.setObjectName("volume")

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Intel R435i Base App"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.btn_photo.setText(_translate("MainWindow", "Picture"))
        self.btn_restart.setText(_translate("MainWindow", "Restart"))
        self.btn_quit.setText(_translate("MainWindow", "Quit"))
        self.checkBox.setText(_translate("MainWindow", "Auto off"))
