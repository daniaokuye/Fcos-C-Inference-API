import os
import time
import ctypes, cv2
import numpy as np
from py_extension.pre_count_lib import FaceCounts
from py_extension.fishEye_lib import FishEye
from ToDB import connectDB


class test(object):
    def __init__(self):
        self.a = 0
        self.s = []
        self.c = []
        self.b = []

    def save(self, scores, classes, boxes):
        self.s.extend([scores.copy()])
        self.c.extend([classes.copy()])
        self.b.extend([boxes.copy()])
        self.a += 1
        if self.a % 200 == 0:
            np.savez('xxx_%d' % self.a, s=self.s, c=self.c, b=self.b)
            self.s = []
            self.c = []
            self.b = []
        print(self.a, "-" * 10)


tt = test()
ratio_w, ratio_h, H, W = (0,) * 4
FC = FaceCounts()
FE = FishEye()
fishObj = connectDB()


def simple_func(data, scale):
    print('*' * 10, data.shape, data.dtype, scale)
    return data


def set_param(r_h, r_w, HH, WW):
    global ratio_w, ratio_h, H, W
    ratio_h, ratio_w, H, W = r_h, r_w, HH, WW


def box_info(scores, classes, boxes):
    # tt.save(scores, classes, boxes)
    info = FC(scores, boxes, classes, ratio_h, ratio_w, H, W)
    # print(ratio_h, ratio_w, H, W)
    # settle results
    list_id = np.array(info['list_id'], dtype=np.float32)

    # list_track
    list_track, list_color, list_track_num = [], [], []
    for i, x in enumerate(info['list_track']):
        list_track_num.append(len(x))
        list_track.extend(x)
        list_color.extend(info['list_color'][i])
    list_track = np.array(list_track, dtype=np.float32)
    list_color = np.array(list_color, dtype=np.float32)
    list_track_num = np.array(list_track_num, dtype=np.float32)

    # list_box
    list_box = np.array(info['list_box'], dtype=np.float32)

    static = np.array([info['entran'], info['pass_by'], info['ratio']], dtype=np.float32)
    support = info['entrance_line']
    support.extend(info['rec'])
    support = np.array(support, dtype=np.float32)

    # print('list_color', list_color, list_color.shape)  # n*3
    # print('list_id', list_id, list_id.shape)
    # print('list_track', list_track, list_track_num, list_track.shape, list_track_num.shape)
    # print('list_box', list_box, list_box.shape)
    # print('entran;pass_by;ratio', static, static.shape)
    # print('entrance_line;rec', support, support.shape)

    return list_id, list_track, list_track_num, list_box, static, support, list_color


def sendToDatabase(img, media_id, frame_id, mac):
    fishObj.push_out(media_id, mac, frame_id, img)
    # fishObj.client.loop_forever()

    # print('=' * 10)
    # print(img.shape, img.dtype, media_id, frame_id, mac)
    # # try:
    # #     cv2.imwrite("aa_.jpg", img.copy())
    # # except Exception as e:
    # #     print('*' * 10, e)
    # # np.savez('xxx_', img)
    # # image = cv2.imread('aa_.jpg')
    # # image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # print('=' * 10)
