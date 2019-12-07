import os
import numpy as np
from pre_count_lib import FaceCounts
from fishEye_lib import FishEye
from ToDB import connectDB
import traceback

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
    info = FC(scores, boxes, classes, ratio_h, ratio_w, H, W)

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

    # support data
    static = np.array([info['entran'], info['pass_by'], info['out_num']], dtype=np.float32)
    support = info['entrance_line']
    support.extend(info['rec'])
    support = np.array(support, dtype=np.float32)
    return list_id, list_track, list_track_num, list_box, static, support, list_color


def sendToDatabase(img, media_id, frame_id, mac):
    try:
        fishObj.push_out(media_id, mac, frame_id, img)
    except Exception as e:
        traceback.print_exc()


if __name__ == '__main__':
    from runProject import test_set

    test_set(True)
    FC.debug = True
    npz = '../build/xxx_%d.npz'
    n = N = 100
    set_param(1.5, 1.6, 1920, 2880)
    while True:
        if not os.path.exists(npz % n) or n / 100 >= 10: break
        print(npz % n)
        data = np.load(npz % n)
        n += N
        data.allow_pickle = True
        scores, classes, boxes = data['s'], data['c'], data['b']
        print(FC.set_MACetc)
        for s, c, b in zip(scores, classes, boxes):
            res = box_info(s, c, b)
