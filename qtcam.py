# -*- coding: utf-8 -*-
import re
import os
import csv
import sys
import cv2
import threading
import numpy as np
import datetime as dt
import tensorflow as tf
from cmd_class import RScam
import multiprocessing as mp
from tensorflow import keras
from temp import Ui_MainWindow
import matplotlib.pyplot as plt
from math_dl.initial import classify
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.animation as animation

a = RScam()
c = classify()
# La clase Ui_MainWindow es la interfaz grafica en codigo
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        # Herencia de las clases de PyQt y definicion de metodo de inicio de la clase
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('img/intel.png'))
        self.setFixedSize(1390, 720)

        # Timer de uso de la camara y de la frecuencia de captura de fotos automatica
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_image)  # Mostrar imagen capturada en los cuadros del main_window
        self.timer_camera.start(int(1000 / 30))             # Captura una imagen cada 1000/n frames

        self.timer_plots = QtCore.QTimer()
        self.timer_plots.timeout.connect(self.update_plots_call)  # Mostrar imagen capturada en los cuadros del main_window

        self.timer_plots1 = QtCore.QTimer()
        self.timer_plots1.timeout.connect(self.update_plots_call_1)

        self.timer_plots2 = QtCore.QTimer()
        self.timer_plots2.timeout.connect(self.update_plots_call_2)

        # Timer de los mensajes de consola
        self.timer_msg = QtCore.QTimer()
        self.timer_msg.timeout.connect(self.get_msg)
        self.timer_msg.start(1000)

        # Centrar la ventana siempre
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        # Conexion de los BOTONES con sus respectivas llamadas de funciones
        # Boton de inicio de camara
        self.btn_start.clicked.connect(lambda : self.start())
        # Boton de reinicio de camara
        self.btn_restart.clicked.connect(lambda: self.restart())
        # Boton de captura de foto manual
        self.btn_photo.clicked.connect(lambda: self.foto())
        # Boton para detener la camara
        self.btn_quit.clicked.connect(lambda: self.quit_btn())
        # Definir valor predeterminado en la caja de conteo de frames
        self.spinBox.setValue(a.distance)
        # Manejador del evento de cambio entre capturas automaticas y manuales
        self.checkBox.stateChanged.connect(lambda: self.auto_change())
        # Manejador del evento de cambio de valor en la caja de intervalo de captura automatica
        self.spinBox.valueChanged.connect(lambda: self.dis_change())
        # Manejador del evento de cambio de texto del textbox de la densidad
        self.density.textChanged.connect(lambda: self.density_on_change())

        # Intenta conseguir la imagen proveniente de la captura
        self.img_rgb = a.rgb_img
        self.img_depth = a.depth_img

        # Parametros globales de los graficos
        # Grafico 0, clasificacion en tiempo real
        self.t1         = None
        self.gData      = None
        self.fig        = None
        self.hl         = None
        self.hl_ax      = None
        self.class_flag = False
        self.t2         = None

        # Grafico 1, #Cavings/Tiempo
        self.nData      = [[0], [0]]
        self.oData1     = [0]
        self.oData2     = [0]
        self.oData3     = [0]
        self.oData4     = [0]
        self.oData5     = [0]
        self.dates      = [0]

        # Figure y Axes del plot historico de classes
        self.t3         = None
        self.fig1       = None
        self.hl1        = None
        self.hl2        = None
        self.hl3        = None
        self.hl4        = None
        self.hl5        = None
        self.hl6        = None
        self.hl_ax1     = None
        self.cont_plots = 0

        # Figure y Axes del plot historico de volumen
        self.t4         = None
        self.fig2       = None
        self.hl7        = None
        self.hl_ax2     = None
        self.vol_hist   = [0]

        # Figure y Axes del plot historico de clases 2
        self.t5 = None
        self.fig3 = None
        self.hl8 = None
        self.hl9 = None
        self.hl10 = None
        self.hl11 = None
        self.hl12 = None
        self.hl13 = None
        self.hl_ax3 = None

        # Datos del plot historico de clases 2
        self.nData1 = [[0], [0]]
        self.oData6 = [0]
        self.oData7 = [0]
        self.oData8 = [0]
        self.oData9 = [0]
        self.oData10 = [0]
        self.dates1 = []

        # Figure y Axes del plot historico de volumen 2
        self.t6 = None
        self.fig4 = None
        self.hl14 = None
        self.hl_ax4 = None
        self.vol_hist1 = [0]

        # Datos temporales
        self.temp_data3 = np.zeros(5)     # Class_his
        self.temp_data4 = 0               # Nums
        self.temp_data5 = 0               # Vol
        self.cont_plots2 = 0
        self.cont_plots3 = 0

        # Valor de la densidad para calculos de volumen
        self.density_val = 2.50

        #Creacion y verificacion del archivo csv de datos de clasificacion
        self.bag  = a.bag
        #data_list = os.listdir('./csv_data/')
        #if self.bag+'.csv' in data_list:
        #    self.bag_mode = 'a'
        #else:
        #    self.bag_mode = 'w'

    # Metodo conectado al boton de inicio de camara
    def start(self):

        # Verifica el status de la camara para el inicio de captura
        if a.cam_status.value == 0:

            # Declara True el estado de reinicio de la camara
            a.restart = True

            # Crea el objeto hilo con el target especifico del metodo main_loop del objeto perteneciente
            # a la clase RSCam y le da inicio
            self.t1 = threading.Thread(target=a.main_loop)
            self.t1.start()

            # Configuracion inicial del plot de clasificacion
            self.gData = [np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0])]
            self.fig = self.MplWidget.canvas.axes.figure
            self.hl_ax = self.fig.axes[0]
            self.hl = self.hl_ax.bar(self.gData[0], self.gData[1], width=1, align='center', color=(0.2, 0.3, 0.8, 0.5))
            self.fig.canvas.draw()
            self.timer_plots.start(int(1000 * 5))

            # Configuracion inicial de los plots historicos
            self.fig1 = self.MplWidget1.canvas.axes.figure
            self.hl_ax1 = self.fig1.axes[0]
            self.hl2 = self.hl_ax1.barh(self.nData[0], self.oData1, height=1, align='center', color=(0.498, 0.772, 0.368, 1))
            self.hl3 = self.hl_ax1.barh(self.nData[0], self.oData2, height=1, align='center', color=(0.415, 0.368, 0.772, 1))
            self.hl4 = self.hl_ax1.barh(self.nData[0], self.oData3, height=1, align='center', color=(0.89, 0.43, 1, 1))
            self.hl5 = self.hl_ax1.barh(self.nData[0], self.oData4, height=1, align='center', color=(0.85, 0.42, 0.00, 1))
            self.hl6 = self.hl_ax1.barh(self.nData[0], self.oData5, height=1, align='center', color=(0.89, 0.74, 0.00, 1))
            self.hl1 = self.hl_ax1.plot(self.nData[1], self.nData[0], 'r--', linewidth=1.5)
            self.hl_ax1.set_title('Cavings/Min')

            self.fig2 = self.MplWidget2.canvas.axes.figure
            self.hl_ax2 = self.fig2.axes[0]
            self.hl7 = self.hl_ax2.barh(0, self.vol_hist, height=1, align='center', color=(0.498, 0.772, 0.368, 1))
            self.hl_ax2.set_title('Volume Estimation/Min')

            self.fig3 = self.MplWidget3.canvas.axes.figure
            self.hl_ax3 = self.fig3.axes[0]
            self.hl8 = self.hl_ax3.barh(self.nData1[0], self.oData6, height=1, align='center', color=(0.498, 0.772, 0.368, 1))
            self.hl9 = self.hl_ax3.barh(self.nData1[0], self.oData7, height=1, align='center', color=(0.415, 0.368, 0.772, 1))
            self.hl10 = self.hl_ax3.barh(self.nData1[0], self.oData8, height=1, align='center', color=(0.89, 0.43, 1, 1))
            self.hl11 = self.hl_ax3.barh(self.nData1[0], self.oData9, height=1, align='center', color=(0.85, 0.42, 0.00, 1))
            self.hl12 = self.hl_ax3.barh(self.nData1[0], self.oData10, height=1, align='center', color=(0.89, 0.74, 0.00, 1))
            self.hl13 = self.hl_ax3.plot(self.nData1[1], self.nData1[0], 'r--', linewidth=1.5)
            self.hl_ax3.set_title('Cavings/Hr')

            self.fig4 = self.MplWidget4.canvas.axes.figure
            self.hl_ax4 = self.fig4.axes[0]
            self.hl14 = self.hl_ax4.barh(0, self.vol_hist1, height=1, align='center', color=(0.498, 0.772, 0.368, 1))
            self.hl_ax4.set_title('Volume Estimation/Hr')

            self.fig1.canvas.draw()
            self.fig2.canvas.draw()
            self.fig3.canvas.draw()
            self.fig4.canvas.draw()

            # Timer del plot 1 y 2
            self.timer_plots1.start(int(1000 * 70))

            # Timer del plot 3 y 4
            self.timer_plots2.start(int(1000 * 3610))

            print("start loop")
            self.class_flag = True
        else:

            print(a.cam_status.value, a.camera_command.value)
            return "Wrong Camera Status!!"


    def density_on_change(self):
        text = self.density.text()
        try:
            if text != "":
                textd = float(text)
                r = re.compile('.*..')
                if r.match(text) and len(text) == 3:
                    if textd > 1.4 and textd < 3.6:
                        self.density_val = textd
                    else:
                        msg = QtWidgets.QMessageBox()
                        msg.setText("The value must be between 1.5 and 3.5!")
                        msg.setWindowTitle("Value Exception")
                        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        msg.show()
                        msg.exec_()
        except Exception:
            msg = QtWidgets.QMessageBox()
            msg.setText("A 2 decimal float number is required please!")
            msg.setWindowTitle("Type Exception")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.show()
            msg.exec_()
            self.density.setText("2.50")


    def update_plots_call(self):
        # Inicio de la animacion de matplotlib
        bars_ani = animation.FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)

        # Hilo de la clasificacion
        self.t2 = threading.Thread(target=self.update_plots, args=(0,))
        self.t2.start()


    #Funcion de actualizacion de todos los plots
    def update_plots(self, num):
        try:
            rgb = a.rgb_img_raw
            depth = a.depth_img_raw
            if rgb.shape[0] == 1080 and depth.shape[0] == 720:

                # Proceso de clasificacion, llamada a la funcion que clasifica con el modelo cargado de formas
                self.gData = c.main_process(rgb, depth)
                if self.gData is not None:
                    # Estimacion de volumen
                    # self.vol_hist.append(int(self.gData[1]))

                    # Datos del numero de cavings sumando las clases
                    self.cont_plots += 1

                    # Datos de fecha y hora
                    date = dt.datetime.now()
                    hora1 = date.strftime('%H:%M')
                    hora2 = date.strftime('%H')
                    dtf = date.isoformat()

                    # Division de los resultados en vectoeres de datos separados
                    #self.temp_data1.append(self.gData[0][0])  # Class_res
                    #self.temp_data2.append(self.gData[0][1])  # Score_res
                    self.temp_data3 += self.gData[0][2]       # Class_his
                    self.temp_data4 += int(self.gData[0][3])  # Nums
                    self.temp_data5 += self.gData[1]          # Vol

                    # Actualizacion del volumen
                    self.volume.setText("Volume: " + str(self.gData[1]/self.density_val)[:4] + " gcm^3/5s")

                    if (self.cont_plots % 12) == 0:
                        csv_m = []
                        self.cont_plots2 += 1
                        self.dates.append(dtf[:10] + ' ' + hora1)
                        csv_m.append(dtf[:10] + ' ' + hora1) # Fecha y hora
                        self.nData[0].append(self.cont_plots2)
                        self.nData[1].append(self.temp_data4)
                        csv_m.append(str(self.temp_data4)) # Numero total de cavings
                        self.oData1.append(int(sum(self.temp_data3[:1])))
                        self.oData2.append(int(sum(self.temp_data3[:2])))
                        self.oData3.append(int(sum(self.temp_data3[:3])))
                        self.oData4.append(int(sum(self.temp_data3[:4])))
                        self.oData5.append(int(sum(self.temp_data3[:5])))
                        self.vol_hist.append(self.temp_data5)
                        csv_m.append(str(self.temp_data5)) # Volumen
                        csv_m.append(str(self.temp_data3)) # Histograma de clases
                        self.temp_data3 = np.zeros(5)  # Class_his
                        self.temp_data4 = 0  # Nums
                        self.temp_data5 = 0  # Vol
                        # Escritura de datos de clasificacion al archivo bag.csv
                        with open('./csv_data/' + self.bag + '_m.csv', 'a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(csv_m)

                    ref1 = 5

                    if (self.cont_plots2 % ref1) == 0 and self.cont_plots2 > 0:
                        self.cont_plots3 += 1
                        self.dates1.append(dtf[:10] + ' ' + hora2)
                        self.nData1[0].append(int(self.cont_plots2/ref1))
                        self.nData1[1].append(sum(self.nData[1][-ref1:]))
                        self.oData6.append(sum(self.oData1[-ref1:]))
                        self.oData7.append(sum(self.oData2[-ref1:]))
                        self.oData8.append(sum(self.oData3[-ref1:]))
                        self.oData9.append(sum(self.oData4[-ref1:]))
                        self.oData10.append(sum(self.oData5[-ref1:]))
                        self.vol_hist1.append(sum(self.vol_hist[-ref1:]))
                        hist_h = [sum(self.oData1[-ref1:]),
                                  sum(self.oData2[-ref1:]),
                                  sum(self.oData3[-ref1:]),
                                  sum(self.oData4[-ref1:]),
                                  sum(self.oData5[-ref1:])]
                        csv_h = []
                        csv_h.append(dtf[:10] + ' ' + hora2) # Fecha y hora
                        csv_h.append(str(sum(self.nData[1][-ref1:]))) # Numero total de cavings
                        csv_h.append(str(sum(self.vol_hist[-ref1:]))) # Volumen
                        csv_h.append(str(hist_h)) # Histograma de clases

                        # Escritura de datos de clasificacion al archivo bag.csv
                        with open('./csv_data/' + self.bag + '_h.csv', 'a', newline='') as csv_file_h:
                            writer_h = csv.writer(csv_file_h)
                            writer_h.writerow(csv_h)


                    self.fig.canvas.axes.clear()
                    self.hl = self.hl_ax.bar(np.array([1, 2, 3, 4, 5]), self.gData[0][2], width=0.5, align='center', color=(0.2, 0.3, 0.8, 0.5))
                    self.hl_ax.set_xlim(0, 6)
                    self.hl_ax.set_xticks([1, 2, 3, 4, 5])
                    self.hl_ax.set_xticklabels(c.class_names, rotation=45)
                    self.hl_ax.set_title('Classification Histogram')
                    self.hl_ax.set_xlabel('Classes')
                    self.hl_ax.set_ylabel('Cavings')
                    self.fig.tight_layout()
                    plt.pause(0.1)
                    self.fig.canvas.draw()

        except Exception:
            print('Exception, could not classify ok')

    def update_plots_call_1(self):
        # Grafico 1 #Cavings/tiempo
        anim_1 = animation.FuncAnimation(self.fig1, self.update_plots_1, interval=50, blit=False)
        anim_2 = animation.FuncAnimation(self.fig2, self.update_plots_2, interval=50, blit=False)

        # Se cra el hilo que va a actualizar el grafico
        self.t3 = threading.Thread(target=self.update_plots_1, args=(0,))
        self.t4 = threading.Thread(target=self.update_plots_2, args=(0,))
        self.t3.start()
        self.t4.start()

    # Actualizacion de plotting de la figura historico de clases
    def update_plots_1(self, num):
        try:
            if len(self.nData[0]) > 1:
                self.fig1 = self.MplWidget1.canvas.axes.figure
                self.hl_ax1 = self.fig1.axes[0]
                if self.gData is not None and np.sum(self.gData[1]) != 0 and len(self.nData[0]) > 30:
                    data  = [self.nData[0][-30:], self.nData[1][-30:], self.dates[-30:]]
                    data1 = [self.nData[0][-30:], self.oData1[-30:],
                             self.nData[0][-30:], self.oData2[-30:],
                             self.nData[0][-30:], self.oData3[-30:],
                             self.nData[0][-30:], self.oData4[-30:],
                             self.nData[0][-30:], self.oData5[-30:]]
                elif self.gData is not None and np.sum(self.gData[1]) != 0 and len(self.nData[0]) <= 30:
                    data  = [self.nData[0], self.nData[1], self.dates]
                    data1 = [self.nData[0], self.oData1,
                             self.nData[0], self.oData2,
                             self.nData[0], self.oData3,
                             self.nData[0], self.oData4,
                             self.nData[0], self.oData5]

                # Plotting
                self.fig1.canvas.axes.clear()
                self.hl6 = self.hl_ax1.barh(data1[8], data1[9], height=0.5, align='center', color=(0.89, 0.74, 0.00, 1), label=c.class_names[4])
                self.hl5 = self.hl_ax1.barh(data1[6], data1[7], height=0.5, align='center', color=(0.85, 0.42, 0.00, 1), label=c.class_names[3])
                self.hl4 = self.hl_ax1.barh(data1[4], data1[5], height=0.5, align='center', color=(0.89, 0.43, 1, 1), label=c.class_names[2])
                self.hl3 = self.hl_ax1.barh(data1[2], data1[3], height=0.5, align='center', color=(0.415, 0.368, 0.772, 1), label=c.class_names[1])
                self.hl2 = self.hl_ax1.barh(data1[0], data1[1], height=0.5, align='center', color=(0.498, 0.772, 0.368, 1), label=c.class_names[0])
                self.hl1 = self.hl_ax1.plot(data[1], data[0], 'r--', linewidth=1.5, label='Total')

                # Embellecer Plot xD
                self.hl_ax1.set_ylim(data[0][0], data[0][-1])
                self.hl_ax1.set_xlim(0, int(np.ceil(max(data1[9]))))
                self.hl_ax1.set_yticks(data[0])
                self.hl_ax1.set_yticklabels(data[2])
                self.hl_ax1.invert_yaxis()
                self.hl_ax1.tick_params(labelsize=8)
                self.hl_ax1.xaxis.set_ticks_position('top')
                self.hl_ax1.grid(color='g', linestyle='-', linewidth='0.5')
                self.hl_ax1.set_title('Cavings/Min')
                self.hl_ax1.legend(loc='upper right')
                self.fig1.tight_layout()
                plt.pause(0.1)
                self.fig1.canvas.draw()

        except Exception:
            print('Fig1 Plotting Error')


    # Funcion de actualizacion de grafico de la figura de historico de volumen
    def update_plots_2(self, num):
        try:
            if len(self.nData[0]) > 1:
                self.fig2 = self.MplWidget2.canvas.axes.figure
                self.hl_ax2 = self.fig2.axes[0]
                if self.gData is not None and np.sum(self.gData[1]) != 0 and len(self.nData[0]) > 30:
                    data  = [self.nData[0][-30:], self.dates[-30:]]
                    data1 = self.vol_hist[-30:]
                elif self.gData is not None and np.sum(self.gData[1]) != 0 and len(self.nData[0]) <= 30:
                    data = [self.nData[0], self.dates]
                    data1 = self.vol_hist

                # Plotting
                self.fig2.canvas.axes.clear()
                self.hl7 = self.hl_ax2.barh(data[0], data1, height=0.5, align='center', color=(0.89, 0.74, 0.00, 1))

                # Embellecer Plot xD
                self.hl_ax2.set_ylim(data[0][0], data[0][-1])
                self.hl_ax2.set_xlim(0, int(np.ceil(max(data1))))
                self.hl_ax2.set_yticks(data[0])
                self.hl_ax2.set_yticklabels(data[1])
                self.hl_ax2.invert_yaxis()
                self.hl_ax2.tick_params(labelsize=8)
                self.hl_ax2.xaxis.set_ticks_position('top')
                self.hl_ax2.grid(color='g', linestyle='-', linewidth='0.5')
                self.hl_ax2.set_title('Volume Estimation/Min')
                self.fig2.tight_layout()
                plt.pause(0.1)
                self.fig2.canvas.draw()

        except Exception:
            print('Fig2 Plotting Error')

    def update_plots_call_2(self):
        anim_3 = animation.FuncAnimation(self.fig3, self.update_plots_3, interval=50, blit=False)
        anim_4 = animation.FuncAnimation(self.fig4, self.update_plots_4, interval=50, blit=False)

        # Se cra el hilo que va a actualizar el grafico
        self.t5 = threading.Thread(target=self.update_plots_3, args=(0,))
        self.t6 = threading.Thread(target=self.update_plots_4, args=(0,))
        self.t5.start()
        self.t6.start()

    def update_plots_3(self):
        try:
            if len(self.oData6) > 1:
                self.fig3 = self.MplWidget3.canvas.axes.figure
                self.hl_ax3 = self.fig3.axes[0]
                if len(self.oData6) <= 30:
                    # Datos de graficacion
                    data  = [self.nData1[0], self.nData1[1], self.dates1]
                    data1 = [self.nData1[0], self.oData6,
                             self.nData1[0], self.oData7,
                             self.nData1[0], self.oData8,
                             self.nData1[0], self.oData9,
                             self.nData1[0], self.oData10]

                elif len(self.oData6) > 30:
                    data = [self.nData1[0][-30:], self.nData[1][-30:], self.dates1[-30:]]
                    data1 = [self.nData1[0][-30:], self.oData1[-30:],
                             self.nData1[0][-30:], self.oData2[-30:],
                             self.nData1[0][-30:], self.oData3[-30:],
                             self.nData1[0][-30:], self.oData4[-30:],
                             self.nData1[0][-30:], self.oData5[-30:]]


                # Plotting
                self.fig3.canvas.axes.clear()
                self.hl12 = self.hl_ax3.barh(data1[8], data1[9], height=0.5, align='center', color=(0.89, 0.74, 0.00, 1), label=c.class_names[4])
                self.hl11 = self.hl_ax3.barh(data1[6], data1[7], height=0.5, align='center', color=(0.85, 0.42, 0.00, 1), label=c.class_names[3])
                self.hl10 = self.hl_ax3.barh(data1[4], data1[5], height=0.5, align='center', color=(0.89, 0.43, 1, 1), label=c.class_names[2])
                self.hl9 = self.hl_ax3.barh(data1[2], data1[3], height=0.5, align='center', color=(0.415, 0.368, 0.772, 1), label=c.class_names[1])
                self.hl8 = self.hl_ax3.barh(data1[0], data1[1], height=0.5, align='center', color=(0.498, 0.772, 0.368, 1), label=c.class_names[0])
                self.hl13 = self.hl_ax3.plot(data[1], data[0], 'r--', linewidth=1.5, label='Total')

                # Embellecer Plot xD
                self.hl_ax3.set_ylim(data[0][0], data[0][-1])
                self.hl_ax3.set_xlim(0, int(np.ceil(max(data1[9]))))
                self.hl_ax3.set_yticks(data[0])
                self.hl_ax3.set_yticklabels(data[2])
                self.hl_ax3.invert_yaxis()
                self.hl_ax3.tick_params(labelsize=8)
                self.hl_ax3.xaxis.set_ticks_position('top')
                self.hl_ax3.grid(color='g', linestyle='-', linewidth='0.5')
                self.hl_ax3.set_title('Cavings/Hr')
                self.hl_ax3.legend(loc='upper right')
                self.fig3.tight_layout()
                plt.pause(0.1)
                self.fig3.canvas.draw()
        except Exception:
            print('Fig3 Plotting Error')

    def update_plots_4(self):
        try:
            if len(self.oData6) > 1:
                self.fig4 = self.MplWidget4.canvas.axes.figure
                self.hl_ax4 = self.fig4.axes[0]
                if len(self.nData1[0]) > 30:
                    data  = [self.nData1[0][-30:], self.dates1[-30:]]
                    data1 = self.vol_hist1[-30:]
                elif len(self.nData1[0]) <= 30:
                    data  = [self.nData1[0], self.dates1]
                    data1 = self.vol_hist1

                # Plotting
                self.fig4.canvas.axes.clear()
                self.hl14 = self.hl_ax4.barh(data[0], data1, height=0.5, align='center', color=(0.89, 0.74, 0.00, 1))

                # Embellecer Plot xD
                self.hl_ax4.set_ylim(data[0][0], data[0][-1])
                self.hl_ax4.set_xlim(0, int(np.ceil(max(data1))))
                self.hl_ax4.set_yticks(data[0])
                self.hl_ax4.set_yticklabels(data[1])
                self.hl_ax4.invert_yaxis()
                self.hl_ax4.tick_params(labelsize=8)
                self.hl_ax4.xaxis.set_ticks_position('top')
                self.hl_ax4.grid(color='g', linestyle='-', linewidth='0.5')
                self.hl_ax4.set_title('Volume Estimation/Hr')
                self.fig4.tight_layout()
                plt.pause(0.1)
                self.fig4.canvas.draw()
        except Exception:
            print('Fig4 Plotting Error')

    # Metodo para mostrar imagen en el main_window
    def show_image(self):
        try:
            self.img_rgb = a.rgb_img
            self.img_depth = a.depth_img

            # Cambia las dimensiones de la imagen para que entren en el espacio
            self.img_rgb = cv2.resize(self.img_rgb, (400, 500))
            heightr, widthr, channelr = self.img_rgb.shape

            self.img_depth = cv2.resize(self.img_depth, (400, 500))
            heightd, widthd, channeld = self.img_depth.shape

        except:
            # En caso de no obtener imagen en la camara, entrega el logo por defecto
            self.img_rgb = cv2.imread('img/1.jpg')
            self.img_rgb = cv2.resize(self.img_rgb, (400, 500))
            heightr, widthr, channelr = self.img_rgb.shape

            self.img_depth = cv2.imread('img/1.jpg')
            self.img_depth = cv2.resize(self.img_depth, (400, 500))
            heightd, widthd, channeld = self.img_depth.shape

        # Creacion de la QImage para mostrar en el main_window a partir del array obtenido arriba
        bytesPerline = 3 * widthr
        self.qImg_rgb = QImage(self.img_rgb.data, widthr, heightr, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qImg_depth = QImage(self.img_depth.data, widthd, heightd, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_rgb.setPixmap(QPixmap.fromImage(self.qImg_rgb))
        self.label_depth.setPixmap(QPixmap.fromImage(self.qImg_depth))

    # Las funciones "restart", "foto", "quit_btn" Entregan al objeto RScam comandos bandera que indican
    def restart(self):
        print("restart")
        a.command.value = 4

    def foto(self):

        print("foto")
        a.command.value = 1

    def quit_btn(self):
        print("quit")
        a.restart = False
        a.command.value = 5
        self.timer_plots.stop()
        self.timer_plots1.stop()
        #self.t1.join()
        self.t2.join()
        self.t3.join()
        self.t4.join()

    def auto_change(self):
        if self.checkBox.isChecked() is False:
            print('manual')
            self.checkBox.setText('Auto off')
            self.checkBox.setChecked(False)
            #a.auto = False
            a.command.value = 3
        else:
            print('auto')
            self.checkBox.setText('Auto on')
            self.checkBox.setChecked(True)
            a.command.value = 2
            a.auto.value = 1

    def dis_change(self):
        a.distance = self.spinBox.value()
        print(a.distance)

    def get_msg(self):
        self.statusbar.showMessage(str(a.msg))

# Clase del hilo que obtiene los mensajes de status de la camara
class msgThread(QThread):
    msg = pyqtSignal(str)

    def __int__(self, parent=None):
        super(msgThread, self).__init__(parent)

    def run(self):
        while True:
            # print('ruuning')
            self.msg.emit(a.msg)

if __name__ == "__main__":
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
