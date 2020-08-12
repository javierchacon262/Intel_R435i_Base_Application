import cv2
import numpy as np
import tensorflow as tf
import yolov3_tf2.models
from tensorflow import keras
from scipy.misc import imresize
import matplotlib.pyplot as plt
from recovery.recoverynet import *
import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import (YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks)


# constantes deep learning
size  = 416
weights = './checkpoints/yolov3_train_50.tf'
mode  = 'fit'
num_classes = 5
learning_rate = 2e-3


class classify:

    def __init__(self):
        self.model_forms = self.form_model_def()
        self.model_vol = self.vol_model_def()
        self.img_rgb = None
        self.img_vol = None
        self.class_names = ['class1', 'class2', 'class3', 'class4', 'class5']
        self.output_forms = None
        self.rgb_res = None
        self.output_vol = None
        self.vol_res = None

    def vol_model_def(self):
        # Cambio de hans dimensiones alrevez y 3ra dimension 2 (antes 4)
        model = UNetL(input_size=(1920, 1080, 2))
        model.load_weights('./checkpoints_vol/Vol_200.tf')
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=False)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def form_model_def(self):
        model = YoloV3(size=size, training=False, classes=num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        model.load_weights(weights)
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=(mode == 'eager_fit'))
        return model


    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


    def box_fusion(self, boxes, scores, classes, nums, wh):
        flags       = np.zeros(nums)
        boxes_sp    = []
        boxes_x     = []
        boxes_y     = []
        boxes_c     = []
        boxes_s     = []
        new_boxes   = []
        new_scores  = []
        new_classes = []
        for i in range(nums):
            x1y1_a = tuple((boxes[i][0:2] * wh).astype(int))
            x2y2_a = tuple((boxes[i][2:4] * wh).astype(int))
            if len(boxes_sp):
                for j in range(len(boxes_sp)):
                    if len(boxes_sp[j]) and not flags[i]:
                        for k in range(len(boxes_sp[j])):
                            idx    = boxes_sp[j][k]
                            x1y1_b = tuple((boxes[idx][0:2] * wh).astype(np.int32))
                            x2y2_b = tuple((boxes[idx][2:4] * wh).astype(np.int32))
                            d1 = {'x1': x1y1_a[0], 'x2': x2y2_a[0], 'y1': x1y1_a[1], 'y2': x2y2_a[1]}
                            d2 = {'x1': x1y1_b[0], 'x2': x2y2_b[0], 'y1': x1y1_b[1], 'y2': x2y2_b[1]}
                            iou = self.get_iou(d1, d2)
                            if iou > 0.1:
                                boxes_sp[j].append(i)
                                boxes_x[j] += [x1y1_a[0], x2y2_a[0]]
                                boxes_y[j] += [x1y1_a[1], x2y2_a[1]]
                                boxes_c[j].append(classes[i])
                                boxes_s[j].append(scores[i])
                                flags[i] = 1
                                break
                    else:
                        if not flags[i]:
                            boxes_sp[j].append(i)
                            boxes_x[j] += [x1y1_a[0], x2y2_a[0]]
                            boxes_y[j] += [x1y1_a[1], x2y2_a[1]]
                            boxes_c[j].append(classes[i])
                            boxes_s[j].append(scores[i])
                            flags[i] = 1

                if not flags[i]:
                    boxes_sp.append([i])
                    boxes_x.append([x1y1_a[0], x2y2_a[0]])
                    boxes_y.append([x1y1_a[1], x2y2_a[1]])
                    boxes_c.append([classes[i]])
                    boxes_s.append([scores[i]])
                    flags[i] = 1
            else:
                boxes_sp.append([i])
                boxes_x.append([x1y1_a[0], x2y2_a[0]])
                boxes_y.append([x1y1_a[1], x2y2_a[1]])
                boxes_c.append([classes[i]])
                boxes_s.append([scores[i]])
                flags[i] = 1

        for i in range(len(boxes_x)):
            xi1 = min(boxes_x[i])
            xi2 = max(boxes_x[i])
            yi1 = min(boxes_y[i])
            yi2 = max(boxes_y[i])
            cla = int(round(np.mean(boxes_c[i])))
            sco = np.mean(boxes_s[i])
            new_boxes.append([xi1, yi1, xi2, yi2])
            new_classes.append(cla)
            new_scores.append(sco)

        nnums = len(new_boxes)

        #img_i = np.copy(self.img_rgb.numpy()[0])
        #for i in range(nnums):
        #    img_i = cv2.rectangle(img=img_i, pt1=(new_boxes[i][0], new_boxes[i][1]), pt2=(new_boxes[i][2], new_boxes[i][3]), color=(255, 0, 0), thickness=2)
        #plt.imshow(img_i)
        #plt.show()
        return new_boxes, new_classes, new_scores, nnums


    def results(self):
        boxes, scores, classes, nums = self.output_forms
        boxes, scores, classes, nums = np.array(boxes[0]).tolist(), np.array(scores[0]).tolist(), np.array(classes[0]).tolist(), int(nums[0])
        class_his = np.zeros(num_classes)
        wh = np.flip(self.img_rgb.shape[1:3])
        #img_i = np.copy(self.img_rgb.numpy()[0])
        #for i in range(nums):
        #    x1y1 = tuple((boxes[i][0:2] * wh).astype(int))
        #    x2y2 = tuple((boxes[i][2:4] * wh).astype(int))
        #    img_i = cv2.rectangle(img=img_i, pt1=(x1y1[0], x1y1[1]), pt2=(x2y2[0], x2y2[1]), color=(255, 0, 0), thickness=2)
        #plt.imshow(img_i)
        #plt.show()
        boxes, classes, scores, nums = self.box_fusion(boxes, scores, classes, nums, wh)

        for i in classes:
            class_his[i] += 1

        return [classes, scores, class_his, nums]


    def draw_labels(self, x, y, class_names):
        img = x.numpy()
        boxes, classes = tf.split(y, (4, 1), axis=-1)
        classes = classes[..., 0]
        wh = np.flip(img.shape[0:2])
        for i in range(len(boxes)):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, class_names[classes[i]],
                              x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              1, (0, 0, 255), 2)
        return img


    def main_process(self, rgb, depth):
        #depthR = imresize(depth, 2.1)

        depthR = cv2.resize(np.float64(depth), (2688, 1512))
        depthR = depthR[207:1287, 297:2217]
        depthR = depthR.reshape([1080, 1920, 1])
        rgb_raw = np.expand_dims(np.mean(np.double(np.copy(rgb)), axis=2), 2)
        self.img_vol = np.concatenate([rgb_raw, depthR], 2)
        Ref_img2 = np.zeros((1920, 1080, 2))
        aux = np.squeeze(self.img_vol[:, :, [0]])
        Ref_img2[:, :, [0]] = np.expand_dims(np.transpose(aux), 2)
        aux = np.squeeze(self.img_vol[:, :, [1]])
        Ref_img2[:, :, [1]] = np.expand_dims(np.transpose(aux), 2)
        #self.img_vol = np.expand_dims(self.img_vol, 0)

        self.img_vol = tf.expand_dims(Ref_img2, 0)

        self.img_rgb = tf.expand_dims(rgb, 0)
        self.img_rgb = dataset.transform_images(self.img_rgb, size)

        # Prediccion con el modelo de formas
        self.output_forms = self.model_forms(self.img_rgb)

        # Estimacion del volumen
        img_res_vol     = self.model_vol.predict(self.img_vol, batch_size=1)[0, :, :, :]
        self.output_vol = 0.5481*(np.sum(img_res_vol)/50000) - 7.9635

        # Analisis de Resultados
        self.rgb_res = self.results()


        return [self.rgb_res, self.output_vol]