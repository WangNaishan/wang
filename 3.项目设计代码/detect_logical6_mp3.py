# -*- coding: utf-8 -*-

import subprocess
import threading
import sys
import cv2
import time
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
# from rknnlite.api import RKNNLite

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sqlite3
from ui.detect_ui import Ui_MainWindow  # 导入detect_ui的界面
# from show_db import MainWindow
from lib.share import shareInfo  # 公共变量名
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from open_sqlite import PatientWindow  # 导入 patient_window 模块
from PyQt5.QtCore import Qt

ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
DATASET = './dataset.txt'
QUANTIZE_ON = True
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640
CLASSES = ("tuberculosis", "")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2]) * 2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 定义一个辅助函数来插入数据到数据库
def insert_detection_data(number_of_boxes, classes):
    conn = sqlite3.connect('detection_database2.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (number_of_boxes,classes)
        VALUES (?, ?)
    ''', (number_of_boxes, classes))

    conn.commit()
    conn.close()


class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, username="", parent=None):
        print(username)
        self.username = username
        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.output_folder = 'output/'
        self.vid_writer = None
        self.patientWindow = None  # 初始化 patientWindow 属性

        # 权重初始文件名
        self.openfile_name_model = None

        # self.rknn = RKNNLite(verbose=False)
        # self.rknn.load_rknn(path="./best.rknn")
        self.init_database()

        self.exit_flag = False  # 退出标志
        self.playing_audio = False
        self.audio_thread = None

    def init_database(self):
        conn = sqlite3.connect('detection_database2.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                number_of_boxes INTEGER NOT NULL,
                classes TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    # 控件绑定相关操作
    def init_slots(self):
        self.ui.pushButton_img.clicked.connect(self.button_image_open)
        self.ui.pushButton_video.clicked.connect(self.button_video_open)
        self.ui.pushButton_camer.clicked.connect(self.button_camera_open)
        self.ui.pushButton_weights.clicked.connect(self.openPatientWindow)
        # self.ui.pushButton_init.clicked.connect(self.model_init)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)

        self.timer_video.timeout.connect(self.show_video_frame)  # 定时器超时，将槽绑定至show_video_frame

    # 查看数据库
    def openPatientWindow(self):
        # 如果 patientWindow 尚未创建，则创建一个新的实例
        if self.patientWindow is None:
            self.patientWindow = PatientWindow()
        self.patientWindow.show()
        self.patientWindow.setWindowModality(Qt.WindowModal)  # 设置模态

    # 图像检测时调用
    def detect(self, img):
        # 使用rknn模型进行初始化运行环境
        ret = self.rknn.init_runtime()
        # Inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 修改图片高度宽度，将输入的图像转换为RGB格式并调整大小为指定的IMG_SIZE
        outputs = self.rknn.inference(inputs=[img])  # 使用rknn进行推理，获取模型输出
        # 对输出进行后处理，将输出reshape并转换为适合模型输入的格式
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]
        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))
        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))  # 转置
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        boxes, classes, scores = yolov5_post_process(input_data)
        if scores is not None and not self.playing_audio:
            self.play_audio_thread()  # 播放音频
        img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            draw(img_1, boxes, scores, classes)  # 如果检结果非空，存入数据库
        # 计算boxes的数量
        number_of_boxes = len(boxes)
        classes_str = "检测到结合杆菌"
        # 插入数据到数据库
        insert_detection_data(number_of_boxes, classes_str)
        print(number_of_boxes, classes_str)
        return img_1

    # 视频检测时调用
    def detect2(self, img):
        # Init runtime environment
        ret = self.rknn.init_runtime()
        # Inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 修改图片高度宽度
        outputs = self.rknn.inference(inputs=[img])
        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]
        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))
        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        boxes, classes, scores = yolov5_post_process(input_data)
        if scores is not None and not self.playing_audio:  # 检测到结核杆菌播放音频
            self.play_audio_thread()
        img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            draw(img_1, boxes, scores, classes)
        return img_1

    # 打开图片并检测
    def button_image_open(self):
        print('button_image_open')
        # name_list = []
        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images",
                                                                "*.jpg;;*.png;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                img = cv2.imread(img_name)
                print("img_name:", img_name)
                info_show = self.detect(img)
                # print(info_show)
                # 获取当前系统时间，作为img文件名
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                file_extension = img_name.split('.')[-1]
                new_filename = now + '.' + file_extension  # 获得文件后缀名
                file_path = self.output_folder + 'img_output/' + new_filename
                cv2.imwrite(file_path, img)
                # 检测信息显示在界面
                # self.ui.textBrowser.setText(info_show)
                # 检测结果显示在界面
                self.result = cv2.cvtColor(info_show, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path

    # 打开视频并检测
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # -------------------------写入视频----------------------------------#
            fps, w, h, save_path = self.set_video_name_and_path()
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            self.timer_video.start(30)  # 以30ms为间隔，启动或重启定时器
            # 进行视频识别时，关闭其他按键点击功能
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)

    # 打开摄像头检测
    def button_camera_open(self):
        print("Open camera to detect")
        # 设置使用的摄像头序号，系统自带为0
        camera_num = 0
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_num)
        # 判断摄像头是否处于打开状态
        bool_open = self.cap.isOpened()
        if not bool_open:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            fps, w, h, save_path = self.set_video_name_and_path()
            fps = 5  # 控制摄像头检测下的fps，Note：保存的视频，播放速度有点快，我只是粗暴的调整了FPS
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.timer_video.start(30)
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)

    # 定义视频帧显示操作
    def show_video_frame(self):
        # name_list = []
        flag, img = self.cap.read()
        if img is not None:
            info_show = self.detect2(img)  # 检测结果写入到原始img上
            # self.vid_writer.write(img) # 检测结果写入视频
            # print(info_show)
            # # 检测信息显示在界面
            # self.ui.textBrowser.setText(info_show)
            show = cv2.resize(info_show, (640, 480))  # 直接将原始img上的检测结果进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 转换适应label的格式
            self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小
        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            self.cap.release()  # 释放video_capture资源
            self.vid_writer.release()  # 释放video_writer资源
            self.ui.label.clear()
            # 视频帧显示期间，禁用其他检测按键功能
            self.ui.pushButton_video.setDisabled(False)
            self.ui.pushButton_img.setDisabled(False)
            self.ui.pushButton_camer.setDisabled(False)
            # 在释放其他资源后释放 cap 和 rknn 资源
            self.rknn.release()
            self.cap.release()

    # 暂停与继续检测
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.ui.pushButton_stop.setText(u'暂停检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'继续检测')

    # 结束视频检测
    def finish_detect(self):
        # self.timer_video.stop()
        self.cap.release()  # 释放video_capture资源
        self.vid_writer.release()  # 释放video_writer资源
        self.ui.label.clear()  # 清空label画布
        # 启动其他检测按键功能
        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            print("Reset stop/begin!")
            self.ui.pushButton_stop.setText(u'暂停/继续')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)

    # 语音播报
    def play_audio_thread(self):
        file_path = '/home/orangepi/Downloads/yushubo/daosheng/zhangzhixuan/mp3.mp3'
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.audio_thread = threading.Thread(target=self.play_audio, args=(file_path,))
            self.audio_thread.start()

    def play_audio(self, file_path):
        subprocess.run(['mpg123', file_path])  # 执行播放音频命令
        print("Playing audio...")
        self.playing_audio = True
        # 模拟音频播放的时间
        # import time
        # time.sleep(5)
        print("Audio finished.")
        self.playing_audio = False

    def set_audio_finished(self):
        self.playing_audio = False


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())
