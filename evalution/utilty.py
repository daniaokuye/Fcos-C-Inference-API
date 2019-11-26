import sys
import os

sys.path.append(os.getcwd() + "/../")

import ctypes, cv2
import numpy as np


def get_box_from_info(info, image_id, cls=0):
    det = []
    for anno in info['up']["annotations"]:
        box = anno["head_bbox"]
        score = anno['tracking_id']
        det.append({
            'image_id': image_id,
            'score': score,
            'bbox': box,
            'category_id': cls
        })
    return det


def get_box_from_divide(boxes, image_id, cls=0):
    det = []
    for box, score in boxes:
        det.append({
            'image_id': image_id,
            'score': score,
            'bbox': box,
            'category_id': cls
        })
    return det


def get_lib(engine_file):
    # kernel = '/home/user/project/retinanet/torch_extension/build/libinfer_2.so'
    kernel = '/home/user/project/run_retina/build/libinfer_2.so'
    print('kernel:', kernel)
    lib = ctypes.cdll.LoadLibrary(kernel)
    info = init(lib, engine_file)
    print(info)
    return lib, info


def init(lib, engine_file, name=b'engine16'):
    # engine_file = b"/home/user/weight/engine16.plan"
    print("init:", engine_file)
    _array = (ctypes.c_int * 10)()
    lib.initialize(engine_file, name, _array)
    info = list(_array)
    return info


def test(lib, img, batch=2):
    name = b'engine16'
    # engine_file = b"/home/user/weight/engine16.plan"
    _array = (ctypes.c_int * 10)()
    dataptr_in = img.ctypes.data_as(ctypes.c_char_p)
    score = np.zeros((batch, 100), dtype=np.float32)
    box = np.zeros((batch, 100, 4), dtype=np.float32)
    cls = np.zeros((batch, 100), dtype=np.float32)
    scoreptr_in = score.ctypes.data_as(ctypes.c_char_p)
    boxptr_in = box.ctypes.data_as(ctypes.c_char_p)
    clsptr_in = cls.ctypes.data_as(ctypes.c_char_p)
    lib.process(dataptr_in, scoreptr_in, boxptr_in, clsptr_in, batch, name)
    return score, box, cls


def x1y1x2y2_x1y1wh(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1 + 1, y2 - y1 + 1
    return np.array([x1, y1, w, h])


def get_blob(im, H, W, flip=False):
    if flip:
        im = im[:, ::-1, :]
    # im = im.astype(np.float32, copy=False)
    im_shape = im.shape

    im_scale = float(H) / float(im_shape[0])
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_shape[1]) > W:
        im_scale = float(W) / float(im_shape[1])
    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
    # pad
    h, w, _ = im_resized.shape
    img = np.pad(im_resized, ((0, H - h), (0, W - w), (0, 0)))
    return img, im_scale
