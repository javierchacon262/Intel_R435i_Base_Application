import multiprocessing as mp
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import datetime
import time
import os, sys
from math import sin, cos, sqrt, atan2, radians
import threading
from scipy.io import savemat


def dir_generate(dir_name):
    """
    :param dir_name: input complete path of the desired directory
    :return: None
    """
    dir_name = str(dir_name)
    if not os.path.exists(dir_name):
        try:
            os.mkdir(dir_name)
        finally:
            pass


class RScam:

    def __init__(self):
        # Create Folders for Data
        if sys.platform == "linux":
            self.root_dir = '/home/pi/RR/'
        else:
            self.root_dir = './'

        folder_list = ('auto', 'fotos', 'foto_log')        # Cambiar ('bag', 'foto_log') por los directorios donde se quieran guardar los .mat

        for folder in folder_list:
            dir_generate(self.root_dir + folder) # Crean los directorios de almacenamiento
        # Create Variables between Processes
        self.Location = mp.Array('d', [0, 0])
        self.Frame_num = mp.Array('i', [0, 0])
        self.take_pic = mp.Value('i', 0)
        self.camera_command = mp.Value('i', 0)
        self.cam_status = mp.Value('i', 0)
        self.bag = self.bag_num()
        jpg_path = "/home/pi/RR/jpg.jpeg"
        if os.path.isfile(jpg_path):
            self.rgb_img = cv2.imread(jpg_path)
            self.depth_img = cv2.imread(jpg_path)
        else:
            self.rgb_img = cv2.imread('img/1.jpg')
            self.depth_img = cv2.imread('img/1.jpg')
        self.rgb_img_raw = self.rgb_img
        self.depth_img_raw = self.depth_img
        self.auto = mp.Value('i', 0)
        self.auto.value = 0
        self.restart = True
        self.command = mp.Value('i', 0)
        self.command.value = 0
        self.distance = 90
        self.contf = mp.Value('i', 0)
        self.contf.value = 0
        self.msg = 'waiting'

    def bag_num(self):  # Esta funcion crea el nombre del archivo .bag donde se guardan las imagenes
        """
        Generate the number of record file MMDD001
        :return:
        """
        num = 1
        now = datetime.datetime.now()
        time.sleep(1)

        try:
            while True:
                file_name = '{:02d}{:02d}_{:03d}'.format(now.month, now.day, num)
                bag_name = 'fotos/Foto_{}.mat'.format(file_name)
                exist = os.path.isfile(bag_name)
                if exist:
                    num += 1
                else:
                    print('current filename:{}'.format(file_name))
                    break
            return file_name
        finally:
            pass

    # Metodo de control del funcionamiento de la camara
    def Camera(self, child_conn, take_pic, camera_status, bag, distance):
        """

        :param child_conn: mp.Pipe for image
        :param take_pic: take pic command, 0 for rest, 1 for take one pic, after taken is 2, log file will turn back to 0
        :param command: 'auto', 'shot', None
        :param camera_status: 0 for rest, 1 for running, 99 for end
        :param bag: bag path /home/pi/bag
        :return:
        """
        print('camera start')
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)
            profile = pipeline.start(config)
            device = profile.get_device()  # get record device

            #########################################################################
            # Max power to the depth sensor
            depth  = device.query_sensors()[0]
            laser_pwr = depth.get_option(rs.option.laser_power)
            print("\n laser power = ", laser_pwr, '\n')
            laser_range = depth.get_option_range(rs.option.laser_power)
            set_laser = laser_range.max
            depth.set_option(rs.option.laser_power, set_laser)
            #########################################################################
            # Other parameters for the camera
            depth.set_option(rs.option.exposure, 715)
            depth.set_option(rs.option.gain, 16)
            print(depth.get_option(rs.option.asic_temperature))
            print(depth.get_option(rs.option.projector_temperature))
            print(depth.get_option(rs.option.depth_units))
            print(depth.get_option(rs.option.stereo_baseline))

            # set frame queue size to max
            sensor = profile.get_device().query_sensors()
            for x in sensor:
                x.set_option(rs.option.frames_queue_size, 32)
            # set auto exposure but process data first
            color_sensor = profile.get_device().query_sensors()[1]
            color_sensor.set_option(rs.option.auto_exposure_priority, True)
            camera_status.value = 1
            self.contf.value = 0
            while camera_status.value != 99:
                # SUPER IMPORTANTE #############################################
                # Inicio del stream de la camara ###############################
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_image = np.asanyarray(depth_color_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap_resize = cv2.resize(depth_image, (400, 250))
                color_cvt = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                color_cvt_2 = cv2.resize(color_cvt, (400, 250))
                raw_rgb = np.asanyarray(color_frame.get_data())
                raw_depth = np.asanyarray(depth_frame.get_data())
                child_conn.send([color_cvt_2, depth_colormap_resize, raw_rgb, raw_depth])
                self.contf.value += 1

                if self.contf.value > 150000:
                    self.contf.value = 0

                if take_pic.value == 1:
                    take_pic.value = 2
                    if self.auto.value:
                        if self.contf.value % distance == 0:
                            savemat('./auto/Foto_' + bag + '_' + str(self.contf.value) + '.mat', {'depth_raw': np.asanyarray(depth_frame.get_data()), 'color_raw': np.asanyarray(color_frame.get_data())})
                            print('taken auto')
                            self.take_pic.value = 2

                    else:
                        savemat('./fotos/Foto_' + bag + '_' + str(self.contf.value) + '.mat', {'depth_raw': np.asanyarray(depth_frame.get_data()), 'color_raw': np.asanyarray(color_frame.get_data())})
                        print('taken')
                        self.command.value = 0
                        self.take_pic.value = 2

            # child_conn.close()
            pipeline.stop()

        except RuntimeError:
            print('run')

        finally:
            print('pipeline closed')
            camera_status.value = 98
            print('camera value', camera_status.value)

    def main_loop(self):
        parent_conn, child_conn = mp.Pipe()
        self.img_thread_status = True
        image_thread = threading.Thread(target=self.image_receiver, args=(parent_conn,))
        image_thread.start()
        while self.restart:
            if self.camera_command.value == 0:
                bag = self.bag # Genera el nombre del archivo .bag
                # le pasan los argumentos al metodo de control de la camara, entre esos el nombre del archivo donde guarda los frames capturados
                cam_process = mp.Process(target=self.Camera, args=(child_conn, self.take_pic, self.camera_command, bag, self.distance))
                cam_process.start()
                self.command_receiver(bag)
                self.msg = 'end one round'
                print(self.distance)
        self.camera_command.value = 0
        self.cam_status.value = 0
        self.img_thread_status = False
        self.msg = "waiting"

    def image_receiver(self, parent_conn):
        while self.img_thread_status:
            try:
                [self.rgb_img, self.depth_img, self.rgb_img_raw, self.depth_img_raw] = parent_conn.recv()
                #print('success')
            except EOFError:
                print('EOF')

        self.rgb_img = cv2.imread("img/1.jpg")
        self.depth_img = cv2.imread("img/1.jpg")
        print("img thread closed")

    def command_receiver(self, bag):
        i = 1
        foto_location = (0, 0)
        while self.camera_command.value != 98:
            present = datetime.datetime.now()
            date = '{},{},{},{}'.format(present.day, present.month, present.year, present.time())
            local_take_pic = False

            if self.take_pic.value == 2:
                logmsg = '{},{}\n'.format(i, date)  # Crea el mensaje para el logfile, lleva registro de las capturas que se hacen
                self.msg = 'Foto {}'.format(i)
                with open('{}foto_log/{}.txt'.format(self.root_dir, bag), 'a') as logfile:  # Abre el .txt del log_file y escribe el registro de la imagen
                    logfile.write(logmsg)
                i += 1
                self.take_pic.value = 0

            if self.take_pic.value in (1, 2):
                continue

            cmd = self.command.value

            if cmd == 2:
                self.auto.value = 1
                local_take_pic = True
            elif cmd == 3:
                self.auto.value = 0
            elif cmd == 1:
                print('take manual')
                local_take_pic = True
            elif cmd == 4 or cmd == 5:
                self.camera_command.value = 99
                self.msg = cmd
                print("close main", self.msg)

            if self.auto.value:
                local_take_pic = True

            if local_take_pic:
                self.take_pic.value = 1

            self.command.value = 0

        self.msg = 'main closed'
        print("main closed")
        self.camera_command.value = 0




if __name__ == '__main__':
    pass