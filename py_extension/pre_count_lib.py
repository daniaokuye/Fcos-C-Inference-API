import sys
import os

# from torch_extension.pre_count.core.load_default_yaml import load_params
# from apis.setting_base import *
# from apis.setting_base_planb import *
# from apis.face_lib import FaceLib
# from torch_extension.pre_count.core.config import merge_priv_cfg_from_file

import logging
# import json
import numpy as np
import copy
import cv2, time

sys.path.insert(0, os.getcwd() + "/../")
cfg_file = os.path.join(os.path.split(__file__)[0], 'face_analysis_config.yaml')
from py_extension.config import cfg_priv, merge_priv_cfg_from_file
from py_extension.fishEye_lib import FishEye
from py_extension.colormap import colormap

# print('-' * 10, cfg_file)
merge_priv_cfg_from_file(cfg_file)
# import requests
# from torch_extension.pre_count.core.imgfunc import *
# from torch_extension.pre_count.airport import FishEye

'''
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_img2dir(img, file_name, path):
    mkdir(path)
    cv2.imencode('.jpg', img)[1].tofile(os.path.join(path, file_name))


def get_pad_img(rect, img, ratio=(0.15, 0.45, 0.3, 0.3)):
    x, y, w, h = rect
    # pad box to square
    if w > h:
        y = y - (w - h) / 2
    else:
        x = x - (h - w) / 2
        w = h
    img_h, img_w, c = img.shape
    new_x1_ = int(x - w * ratio[2])
    new_y1_ = int(y - w * ratio[0])
    new_x2_ = int(x + w * (1 + ratio[3]))
    new_y2_ = int(y + w * (1 + ratio[1]))
    new_x1, padx1 = [new_x1_, 0] if new_x1_ > 0 else [0, -new_x1_]
    new_y1, pady1 = [new_y1_, 0] if new_y1_ > 0 else [0, -new_y1_]
    new_x2, padx2 = [new_x2_, 0] if new_x2_ < img_w else [img_w, new_x2_ - img_w]
    new_y2, pady2 = [new_y2_, 0] if new_y2_ < img_h else [img_h, new_y2_ - img_h]
    face_img = img[new_y1:new_y2, new_x1:new_x2]
    face_img = cv2.copyMakeBorder(face_img, pady1, pady2, padx1, padx2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return face_img


def get_org_img(img):
    img_h, img_w, c = img.shape
    face_img = img[int(img_h * 0.15):int(img_h * 0.65), int(img_h * 0.25):int(img_h * 0.75)]
    return face_img
'''


class FaceCounts(object):
    def __init__(self):
        super().__init__()
        self.FishEye = FishEye()
        # files = b"/home/user/weight/int8_%d_%d.plan"
        # name = b'engine_%d_%d'
        # self.FishEye.model_0(files % m_param)
        # self.FaceLib = FaceLib()
        # self.FaceLib.camera_id = cam_id
        # format_info = '%(filename)s - %(lineno)s - %(message)s' if cfg_priv.LOG == 'logging.DEBUG' else ''
        # logging.basicConfig(level=eval(cfg_priv.LOG), format=format_info)
        # print('cfg file:', cfg_priv.GLOBAL.VISION_PROJECT_ROOT + "/cfgs/face_analysis_config_bak{}.yaml".format(0))
        # load_params(cam_id)

        # if cfg_priv.OTHER.API_PLAN_A:
        #     self.thread_a = myThreadA(cam_id, [1080, 1920])
        #     self.thread_a.setDaemon(True)
        #     self.thread_a.start()
        # else:
        #     self.thread_b = myThreadB(cam_id, [1080, 1920])
        #     self.thread_b.setDaemon(True)
        #     self.thread_b.start()

        self.width = 1920
        self.pad = 0  # int(self.width * 0.1) // 2
        self.tracks = dict()
        self.tracks['up'] = dict()
        # self.tracks['down'] = dict()
        self.in_num = 0
        self.out_num = 0
        self.pass_num = 0
        self.ratio = 0
        self.out_info = {'list_track': [], 'list_box': [], 'list_id': []}
        self.curID = 0
        assert cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != "None", \
            "cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != None"
        assert len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0, \
            "len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0"
        assert cfg_priv.OTHER.COUNT_MODE == True, \
            "cfg_priv.OTHER.COUNT_MODE == True"
        self.redefine = False
        self.rect = [[0, 0], [1, 1]]
        self.entrance_line = [[2, 2], [4, 4]]

    def find_rect(self, points, shape, pad):
        p_array = np.array(points)
        x = p_array[:, 0]
        y = p_array[:, 1]

        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)
        xmin, xmax = xmin, xmax
        ymin, ymax = ymin, ymax

        if xmin < 0:
            xmin = 0
        if xmax >= shape[1]:
            xmax = shape[1]
        if ymin < 0:
            ymin = 0
        if ymax >= shape[0]:
            ymax = shape[0]
        rect_out = [[xmin + pad, ymin], [xmax + pad, ymax]]
        return rect_out

    def find_direct(self, rect):
        x = int((rect[0][0] + rect[1][0]) / 2)
        y1 = rect[0][1] + int((rect[1][1] - rect[0][1]) / 4)
        y2 = rect[1][1] - int((rect[1][1] - rect[0][1]) / 4)
        direction_out = [[x, y1], [x, y2]]
        return direction_out

    def generalequation(self, first_x, first_y, second_x, second_y):
        coeff_a = second_y - first_y
        coeff_b = first_x - second_x
        coeff_c = second_x * first_y - first_x * second_y
        return coeff_a, coeff_b, coeff_c

    def cross_point(self, line1, line2):
        x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
        x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
        coeff_a1, coeff_b1, coeff_c1 = self.generalequation(x1, y1, x2, y2)
        coeff_a2, coeff_b2, coeff_c2 = self.generalequation(x3, y3, x4, y4)
        m = coeff_a1 * coeff_b2 - coeff_a2 * coeff_b1
        if m == 0:
            x = None
            y = None
            return [x, y]
        else:
            x = (coeff_c2 * coeff_b1 - coeff_c1 * coeff_b2) / m
            y = (coeff_c1 * coeff_a2 - coeff_c2 * coeff_a1) / m
        return [int(x), int(y)]

    def _rect_inter_inner(self, x1, x2):
        n1 = x1.shape[0] - 1
        n2 = x2.shape[0] - 1
        X1 = np.c_[x1[:-1], x1[1:]]
        X2 = np.c_[x2[:-1], x2[1:]]
        S1 = np.tile(X1.min(axis=1), (n2, 1)).T
        S2 = np.tile(X2.max(axis=1), (n1, 1))
        S3 = np.tile(X1.max(axis=1), (n2, 1)).T
        S4 = np.tile(X2.min(axis=1), (n1, 1))
        return S1, S2, S3, S4

    def _rectangle_intersection_(self, x1, y1, x2, y2):
        S1, S2, S3, S4 = self._rect_inter_inner(x1, x2)
        S5, S6, S7, S8 = self._rect_inter_inner(y1, y2)

        C1 = np.less_equal(S1, S2)
        C2 = np.greater_equal(S3, S4)
        C3 = np.less_equal(S5, S6)
        C4 = np.greater_equal(S7, S8)

        ii, jj = np.nonzero(C1 & C2 & C3 & C4)
        return ii, jj

    def intersection(self, x1, y1, x2, y2):
        ii, jj = self._rectangle_intersection_(x1, y1, x2, y2)
        n = len(ii)

        dxy1 = np.diff(np.c_[x1, y1], axis=0)
        dxy2 = np.diff(np.c_[x2, y2], axis=0)

        T = np.zeros((4, n))
        AA = np.zeros((4, 4, n))
        AA[0:2, 2, :] = -1
        AA[2:4, 3, :] = -1
        AA[0::2, 0, :] = dxy1[ii, :].T
        AA[1::2, 1, :] = dxy2[jj, :].T

        BB = np.zeros((4, n))
        BB[0, :] = -x1[ii].ravel()
        BB[1, :] = -x2[jj].ravel()
        BB[2, :] = -y1[ii].ravel()
        BB[3, :] = -y2[jj].ravel()

        for i in range(n):
            try:
                T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
            except:
                T[:, i] = np.NaN

        in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

        xy0 = T[2:, in_range]
        xy0 = xy0.T
        return xy0[:, 0], xy0[:, 1]

    def get_tracks(self, img_data, current_id, max_lost_frames=5):
        # if len(img_data['up']["delete_tracking_id"]):
        #     print("up", '- ' * 10, img_data['up']["delete_tracking_id"])
        # if len(img_data['down']["delete_tracking_id"]):
        #     print('dn', '- ' * 10, img_data['down']["delete_tracking_id"])
        for person in img_data['up']["annotations"]:
            track_id = person["tracking_id"]
            global_id = person["global_id"]
            box_xywh = person["head_bbox"]
            position = [int(box_xywh[0] + box_xywh[2] / 2), int(box_xywh[1] + box_xywh[3] / 2)]

            if track_id in self.tracks['up']:
                self.tracks['up'][track_id]['boxes'] = box_xywh
                self.tracks['up'][track_id]['track'].append(position)
                self.tracks['up'][track_id]['latest_frame'] = current_id
            else:
                self.tracks['up'][track_id] = dict()
                self.tracks['up'][track_id]['boxes'] = box_xywh
                self.tracks['up'][track_id]['track'] = []
                self.tracks['up'][track_id]['track'].append(position)
                self.tracks['up'][track_id]['draw_id'] = global_id
                self.tracks['up'][track_id]['status'] = True
                self.tracks['up'][track_id]['draw'] = True
                self.tracks['up'][track_id]['start_frame'] = current_id
                self.tracks['up'][track_id]['latest_frame'] = current_id

        for track_id in self.tracks['up'].keys():
            if current_id - self.tracks['up'][track_id]['latest_frame'] > max_lost_frames:
                img_data['up']["delete_tracking_id"].append(track_id)
            if track_id in img_data['up']["delete_tracking_id"]:
                # if track_id == 4:
                #     print()
                self.tracks['up'][track_id]['status'] = False

    def clear_all(self):
        self.in_num = 0
        self.out_num = 0
        self.pass_num = 0
        self.ratio = 0

    def smart_judge(self, key, track_id):
        # can a stop point of a line being a start point of another line?
        dead_id = self.tracks[key][track_id]
        up_and_down_frames = 2

        for instance in self.tracks[key].values():
            if 0 <= instance['start_frame'] - dead_id['latest_frame'] < 2 * up_and_down_frames:
                if not (len(instance['track']) or len(dead_id['track'])): continue
                dis = np.array(instance['track'][0]) - np.array(dead_id['track'][-1])
                if np.sum(np.abs(dis) < 10) == 2:
                    # connect them
                    dead_id['track'].extend(instance['track'])
                    instance['track'] = dead_id['track']
                    instance['start_frame'] = dead_id['start_frame']
                    self.tracks[key].pop(track_id)
                    return True
        return False

    def count_num(self, rect_area, entra_line):
        tracks_tmp_up = copy.deepcopy(self.tracks['up'])
        # tracks_tmp_down = copy.deepcopy(self.tracks['down'])

        track_ids = tracks_tmp_up.keys()
        pop_ids = []
        for track_id in track_ids:
            occur = (self.curID == self.tracks['up'][track_id]['latest_frame'])
            if self.tracks['up'][track_id]['status']:
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks['up'][track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks['up'][track_id])
            else:
                is_combine = self.smart_judge('up', track_id)
                if is_combine: continue
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks['up'][track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks['up'][track_id])
                # todo:does not work
                # if len(self.tracks['up'][track_id]['track']) > 10:
                #     in_tmp, out_tmp, pass_tmp = self.deter_in_out(self.tracks['up'][track_id]['track'], rect_area,
                #                                                   entra_line)
                #     self.in_num += in_tmp
                #     self.out_num += out_tmp
                #     self.pass_num += pass_tmp
                #     if self.pass_num == 0:
                #         self.ratio = 0
                #     else:
                #         self.ratio = self.in_num * 1.0 / self.pass_num * 100
                pop_ids.append(track_id)
        for track_id in pop_ids:
            self.tracks['up'].pop(track_id)

    def is_in_area(self, point, area):
        flag = cv2.pointPolygonTest(area, point, False)
        if flag == 1:
            return True
        else:
            return False

    def transfer_np(self, points):
        np_points = np.array(points)
        x_points = np_points[:, 0]
        y_points = np_points[:, 1]
        return x_points, y_points

    def cross_entra_line(self, v_track, v_entra_line):
        x_track, y_track = self.transfer_np(v_track)
        x_entra_line, y_entra_line = self.transfer_np(v_entra_line)
        ret_x, ret_y = self.intersection(x_entra_line, y_entra_line, x_track, y_track)
        if len(ret_x) > 0:
            return True
        return False

    def deter_in_out_new(self, track, v_rect, entra_line):
        rect_area = np.array([[v_rect[0][0], v_rect[0][1]], [v_rect[1][0], v_rect[0][1]], [v_rect[1][0], v_rect[1][1]],
                              [v_rect[0][0], v_rect[1][1]]])
        out = []
        for t in track:
            out.append(self.is_in_area(t, rect_area))
        # todo:check multi line have a location not far away
        out = np.array(out)
        if np.sum(out) / len(out) > 0.5:
            is_pass = 1

    def deter_in_out(self, track, v_rect, entra_line):
        start_point = tuple(track[0])
        end_point = tuple(track[-1])
        rect_area = np.array([[v_rect[0][0], v_rect[0][1]], [v_rect[1][0], v_rect[0][1]], [v_rect[1][0], v_rect[1][1]],
                              [v_rect[0][0], v_rect[1][1]]])

        s_flag = self.is_in_area(start_point, rect_area)
        e_flag = self.is_in_area(end_point, rect_area)
        # print("strat:", s_flag)
        # print("strat:", s_flag)

        if s_flag is True and e_flag is False:
            if self.cross_entra_line(track, entra_line):
                # print("logging: in")
                return 1, 0, 1

        if s_flag is False and e_flag is True:
            if self.cross_entra_line(track, entra_line):
                # print("logging: out")
                return 0, 1, 0

        if s_flag is False and e_flag is False:
            # print("logging: non")
            return 0, 0, 0

        if s_flag is True and e_flag is True:
            # print("logging: pass")
            return 0, 0, 1

        return 0, 0, 0

    def draw_scope(self, rec, entrance_line):
        self.out_info['rec'] = rec
        self.out_info['entrance_line'] = entrance_line
        # if cfg_priv.OTHER.DRAW_ROI:
        #     cv2.rectangle(image, (rec[0][0], rec[0][1]), (rec[1][0], rec[1][1]),
        #                   (18, 127, 15), thickness=5)
        # # if cfg_priv.OTHER.DRAW_DIRECTION:
        # #     cv2.arrowedLine(image['up'], (direct[0][0], direct[0][1]), (direct[1][0], direct[1][1]), (255, 227, 218),
        # #                     thickness=5,
        # #                     shift=0, tipLength=0.2)
        #
        # if entrance_line == "None" or entrance_line is None:
        #     pass
        # else:
        #     if cfg_priv.OTHER.DRAW_ENTRA_LINE:
        #         cv2.line(image, (entrance_line[0][0], entrance_line[0][1]),
        #                  (entrance_line[1][0], entrance_line[1][1]),
        #                  (145, 255, 222), 5)
        # # return image

    def draw_num(self):
        self.out_info['pass_by'] = self.pass_num
        self.out_info['entran'] = self.in_num
        self.out_info['ratio'] = round(self.ratio, 2)
        # if self.pass_num != 0:round(self.ratio, 2)
        #     chinese_str_list = ["过店人次: " + str(self.pass_num), "进店人次: " + str(self.in_num),
        #                         "进店率: " + str(round(self.ratio, 2)) + "%"]
        # else:
        #     chinese_str_list = ["过店人次: " + str(self.pass_num), "进店人次: " + str(self.in_num),
        #                         "进店率: --"]
        # image = vis_inout_result(image, chinese_str_list)
        #
        # return image

    def draw_track(self, track):
        # H, W  =  shape
        # id = track['draw_id']
        color_map = colormap()
        color = int(color_map[track['draw_id'] % 79][0]), int(color_map[track['draw_id'] % 79][1]), int(
            color_map[track['draw_id'] % 79][2])

        if track['draw']:
            if cfg_priv.OTHER.COUNT_DRAW_LESS:
                new_track = track['track'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
            else:
                new_track = track['track']
            self.out_info['list_track'].append(new_track)
            self.out_info['list_box'].append(track['boxes'])
            self.out_info['list_id'].append(track['draw_id'])
            self.out_info['list_color'].append(color)
            # for j in range(len(new_track)):
            #     if cfg_priv.OTHER.DRAW_TRACK:
            #         cv2.circle(img_draw, (new_track[j][0], new_track[j][1]), 1, color, 0)
            #     if cfg_priv.OTHER.DRAW_HEAD:
            #         if j == len(new_track) - 1:
            #             cv2.rectangle(img_draw, (track['boxes'][0], track['boxes'][1]),
            #                           (track['boxes'][0] + track['boxes'][2],
            #                            track['boxes'][1] + track['boxes'][3]),
            #                           color, thickness=style)
            #             cv2.putText(img_draw, str(id), (int(track['boxes'][0]) + 6, int(track['boxes'][1]) + 6),
            #                         cv2.FONT_HERSHEY_SIMPLEX, max(1.0, style * 0.4), color,
            #                         max(1, int(style * 0.6)))
            #
            #     if cfg_priv.OTHER.DRAW_TRACK:
            #         if j != 0:
            #             # if (new_track[j][1] < H // 2) != (new_track[j - 1][1] < H // 2): continue
            #             if abs(new_track[j][0] - new_track[j - 1][0]) > W // 2: continue
            #             cv2.line(img_draw, (new_track[j - 1][0], new_track[j - 1][1]),
            #                      (new_track[j][0], new_track[j][1]), color, thickness=style)
            # else:
            #     for j in range(len(track['track'])):
            #         if cfg_priv.OTHER.DRAW_TRACK:
            #             cv2.circle(img_draw, (track['track'][j][0], track['track'][j][1]), 1, color, 0)
            #         if cfg_priv.OTHER.DRAW_HEAD:
            #             if j == len(track['track']) - 1:
            #                 cv2.rectangle(img_draw, (track['boxes'][0], track['boxes'][1]),
            #                               (
            #                                   track['boxes'][0] + track['boxes'][2],
            #                                   track['boxes'][1] + track['boxes'][3]),
            #                               color, thickness=5)
            #                 cv2.putText(img_draw, str(id), (track['boxes'][0] + 6, track['boxes'][1] + 6),
            #                             cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
            #         if cfg_priv.OTHER.DRAW_TRACK:
            #             if j != 0:
            #                 if (track['track'][j][1] < H // 2) != (track['track'][j - 1][1] < H // 2): continue
            #                 cv2.line(img_draw, (track['track'][j - 1][0], track['track'][j - 1][1]),
            #                          (track['track'][j][0], track['track'][j][1]), color, thickness=5)
        # return img_draw

    # def return_info(self):
    #     return self.FaceLib.single_img_info_dict

    def return_count(self):
        return self.in_num, self.out_num, self.pass_num, self.ratio

    # def strip_pad(self, v_img, v_pad):
    #     return v_img[:, v_pad:v_pad + self.width]

    def get_line_offset(self, v_line, v_pad):
        return [[v_line[0][0] + v_pad, v_line[0][1]], [v_line[1][0] + v_pad, v_line[1][1]]]

    def __call__(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        self.curID += 1
        self.out_info = {'list_track': [], 'list_box': [], 'list_id': [], 'list_color': [],
                         'entran': 0, 'pass_by': 0, 'ratio': 0}
        if self.redefine:
            self.redefine = False
            if cfg_priv.BUSS.COUNT.ROI_AREA['default'] == "None" or cfg_priv.BUSS.COUNT.ROI_AREA[
                'default'] is None or len(cfg_priv.BUSS.COUNT.ROI_AREA['default']) == 0:
                rect = ([0 + self.pad, 0], [W + self.pad, H // 2])
            else:
                rect = self.find_rect(cfg_priv.BUSS.COUNT.ROI_AREA['default'], (H, W), self.pad)
            entrance_line = self.get_line_offset(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'], self.pad)
        else:
            rect, entrance_line = self.rect, self.entrance_line

        ##################################
        single_img_info_dict = self.FishEye(scores, boxes, classes, ratio_h, ratio_w, H, W)
        if cfg_priv.OTHER.COUNT_DRAW:
            self.draw_scope(rect, entrance_line)
        try:
            self.get_tracks(single_img_info_dict, self.curID)
            self.count_num(rect, entrance_line)
        except Exception as e:
            print("error occur:", e, '*' * 10, single_img_info_dict, '*' * 10)
        if cfg_priv.OTHER.COUNT_DRAW:
            self.draw_num()
        return self.out_info


if __name__ == '__main__':
    npz = '../build/xxx_%d00.npz'
    n = 2
    while True:
        if not os.path.exists(npz % n) or n >= 10:
            break
        print(npz % n)
        data = np.load(npz % n)
        n += 2
        data.allow_pickle = True
        scores, classes, boxes = data['s'], data['c'], data['b']
        fc = FaceCounts()
        i = 0
        for s, c, b in zip(scores, classes, boxes):
            res = fc(s, b, c, 1.5, 1.6, 1920, 2880)
            print('*', end='')
