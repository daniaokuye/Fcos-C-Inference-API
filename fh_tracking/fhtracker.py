from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import logging


class HFtracker(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # tracker parms
        iou_cost_weight = 4
        cost_th = 1
        max_mismatch_times = 30
        self.lib = cdll.LoadLibrary('/home/user/project/retinanet/torch_extension/fh_tracking/libHF.so')
        self.lib.new_tracker.restype = c_void_p
        self.lib.new_tracker.argtypes = [c_float, c_float, c_int]
        self.obj = self.lib.new_tracker(iou_cost_weight, cost_th, max_mismatch_times)
        self.lib.tracking_Frame_Hungarian.restype = ndpointer(dtype=c_int, shape=(500,))

    def tracking_Frame_Hungarian(self, detection_rects, img_w, img_h, cut_lines):
        line_num = len(cut_lines)
        cut_lines = (c_int * line_num)(*cut_lines)
        if detection_rects is None:
            detection_rects = 0
            box_num = 0
            self.lib.tracking_Frame_Hungarian.argtypes = \
                [c_void_p, c_int, c_int, c_int, c_int, (c_int * line_num), c_int]
            # self.lib.tracking_Frame_Hungarian(self.obj, detection_rects, box_num, img_w, img_h)
            # return [], []
        else:
            detection_rects[:, 2:4] -= detection_rects[:, 0:2]
            detection_rects[:, 2:4] += 1e-5  # in case equal to 0
            detection_rects = list(detection_rects.flatten())
            box_num = len(detection_rects)
            self.lib.tracking_Frame_Hungarian.argtypes = \
                [c_void_p, (c_int * box_num), c_int, c_int, c_int, (c_int * line_num), c_int]
            detection_rects = (c_int * box_num)(*detection_rects)
            box_num = int(box_num / 4)

        tracking_all_result = self.lib.tracking_Frame_Hungarian(self.obj, detection_rects, box_num,
                                                                img_w, img_h, cut_lines, line_num)
        tracking_all_result = tracking_all_result.astype(np.int32).tolist()
        # print('all_result', tracking_all_result)
        tracking_result = tracking_all_result[0:box_num]
        delete_tracking_id_num = tracking_all_result[box_num]
        delete_tracking_id = tracking_all_result[box_num + 1:box_num + 1 + delete_tracking_id_num]
        # print('tracking_result', tracking_result)
        # print('delete_tracking_id', delete_tracking_id)
        return tracking_result, delete_tracking_id
