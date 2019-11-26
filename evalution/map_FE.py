import sys
import os

sys.path.append(os.getcwd() + "/../")

import time
import json
import ctypes, cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utilty import *
from test_infer import test_simple_infer
from evalution.VOCstyle_xml import *
from py_extension.ut import Profiler
from py_extension.airport_untities import *
from py_extension.fishEye_lib import FishEye


# https://zhuanlan.zhihu.com/p/60834912
# https://arleyzhang.github.io/articles/c521a01c/
#################################################################
#################################################################
# using a images, collect all three parts into one results
def prepare_map_cal(m_param, file, batch=1, visual=False):
    fisheye = FishEye()
    fisheye.nms_thres = 0.0
    fisheye.Evaluate = True
    files = b"/home/user/weight/int8_%d_%d.plan"
    fisheye.model_0(files % m_param)
    print(files % m_param)
    profiler = Profiler(['read', 'engine'])
    fisheye.profiler = profiler
    fisheye.overlap = True
    ii = 0
    imgs = []
    detections = []
    for image_id, file_name in file:
        ii += 1
        profiler.start('read')
        # file_name = os.path.join(file, name)
        if not os.path.exists(file_name):
            print("file <{}> and id {} does not exist".format(file_name, image_id))
            continue
        img = cv2.imread(file_name)
        imgs.append(img)
        if len(imgs) != batch:
            continue
        img, imgs = np.stack(imgs), []
        profiler.stop('read')
        profiler.start('engine')
        # boxes = fisheye.engine(img, ii, visual=visual)
        # next(boxes)
        # def __call__(self, scores, boxes, classes, ratio_h, ratio_w, H, W):

        info = fisheye(img, ii, visual=visual)
        det = get_box_from_info(info, image_id)
        detections.extend(det)
        # next(boxes)
        profiler.stop('engine')
    np.savez('int8_%d_%d' % m_param, k=detections)
    fisheye.free_Ins()
    del fisheye


# using separate images
def prepare_map_cal_divide(m_param, file, batch=1, savedCls=0, visual=False):
    nms_thres = 0.05
    # fisheye.Evaluate = True
    files = b"/home/user/project/run_retina/weights/fcos_int8_%d_%d.plan" % m_param
    # files = b"/home/user/weight/newint8.plan"
    # files = b'/home/user/weight/fcos_int8_640x1280.plan'
    FaceDet, lib_info = get_lib(files)
    print(files)
    profiler = Profiler(['read', 'engine'])
    # fisheye.profiler = profiler
    # fisheye.overlap = True
    ii = 0
    Mh, Mw = lib_info[:2]

    detections = []
    for image_id, file_name in file:
        ii += 1
        profiler.start('read')
        if not os.path.exists(file_name):
            print("file <{}> and id {} does not exist".format(file_name, image_id))
            continue
        imgs = cv2.imread(file_name)

        img, ratio = get_blob(imgs, Mh, Mw)
        profiler.stop('read')

        img = img.astype(np.float32)
        img = img - np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
        img = img[np.newaxis]
        img = img.transpose(0, 3, 1, 2)

        profiler.start('engine')
        scores, boxes, classes = test(FaceDet, img.copy(), batch=1)
        profiler.stop('engine')
        idx = (scores > nms_thres) * (classes == 1)
        # if not idx.sum():
        #     continue
        boxes[..., 0::2] /= ratio
        boxes[..., 1::2] /= ratio
        box = [[x1y1x2y2_x1y1wh(b), s] for b, s in zip(boxes[idx], scores[idx])]
        ################################################
        # h, w, c = imgs.shape
        # for b in boxes[idx]:
        #     x1, y1, x2, y2 = b.round().astype(np.int)
        #     cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.imwrite('/home/user/aa_%d.jpg' % ii, imgs)
        ################################################

        det = get_box_from_divide(box, image_id, savedCls)
        detections.extend(det)
        show_eclipse(profiler, batch)
    saveFile = 'int8_%d_%d_S' % m_param
    print("*" * 10, saveFile, "*" * 10)
    np.savez(saveFile, k=detections)


def evalue_map_cal(npzFile, coco):
    print('\n*-* test %s now! *-*\n' % npzFile)
    info = np.load(npzFile)
    info.allow_pickle = True
    detections = info['k'].tolist()
    if len(detections):
        # Evaluate model on dataset
        if 'annotations' in coco.dataset:
            print('Evaluating model...')
            coco_pred = coco.loadRes(detections)
            coco_eval = COCOeval(coco, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
    else:
        print('No detections!')


def split_eval():
    model = ((640, 1280), (896, 896 * 2),)
    # files  = ('/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/CH_CLIP_20190816-173844/perimeter_415.jpg',)
    jpg_path = '/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/'
    json_file = '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/person.json'
    with open(json_file)as f:
        data = json.load(f)
    coco = COCO(json_file)
    files = []
    for id in coco.imgs.keys():
        image = coco.loadImgs(id)[0]['file_name']
        files.append((id, '{}/{}'.format(jpg_path, image)))
    for mm in model:
        prepare_map_cal(mm, files)
        evalue_map_cal('int8_%d_%d.npz' % mm, coco)


def split_eval_divide(this_model, mode=1):
    # model = [(w, w * s) for w in [640, 768, 896] for s in [1, 2]]
    model = (this_model,)  # (896, 896 * 2),
    print(model)
    # files  = ('/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/CH_CLIP_20190816-173844/perimeter_415.jpg',)
    # jpg_path = '/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/'
    # json_file = '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/person_S.json'
    # clsTarget = 0

    jpg_path = '/home1/datasets/coco/images/val2017'
    json_file = '/home1/datasets/coco/annotations/person_keypoints_val2017.json'
    clsTarget = 1
    with open(json_file)as f:
        data = json.load(f)
    coco = COCO(json_file)
    files = []
    for id in coco.imgs.keys():
        image = coco.loadImgs(id)[0]['file_name']
        files.append((id, '{}/{}'.format(jpg_path, image)))
    for mm in model:
        print("*" * 10, 'int8_%d_%d_S.npz' % mm)
        if mode == 1:
            prepare_map_cal_divide(mm, files, savedCls=clsTarget)
        else:
            evalue_map_cal('int8_%d_%d_S.npz' % mm, coco)
        # evalue_map_cal('int8_640_1280_S_new.npz', coco)


def show_eclipse(profiler, batch):
    profiler.bump('infer')
    # print(self.profiler.totals['infer'])
    msg = ''
    if (profiler.totals['infer'] > 10):
        for name in profiler.names:
            if not name: continue
            msg += '({}: {:.3f}s) '.format(name, profiler.means[name])
        msg += ', {:.1f} im/s'.format(batch / profiler.means['infer'])
        print(msg, flush=True)
        profiler.reset()


if __name__ == '__main__':
    params = [eval(i) for i in sys.argv[1:]]
    if (len(params) != 3): print("argv should have 3 params!")
    if params[0] == 1:
        print("for eclipse:")
        test_simple_infer((params[1], params[2]))
    if params[0] == 2:
        print("for precision:")
        split_eval_divide((params[1], params[2]), mode=1)
    if params[0] == 3:
        print("for precision:")
        split_eval_divide((params[1], params[2]), mode=2)
