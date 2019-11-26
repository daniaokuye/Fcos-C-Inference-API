import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + "/../../")

import time
import cv2
import numpy as np
from torch_extension.ut import Profiler

from torch_extension.pre_count.airport_untities import *
from torch_extension.fisheye.fishEye_lib import FishEye
from torch_extension.fisheye.ap_cal import plot_main
from torch_extension.fisheye.VOCstyle_xml import *
from torch_extension.pre_count.airport_untities import npbbox_iou


# https://zhuanlan.zhihu.com/p/60834912
# https://arleyzhang.github.io/articles/c521a01c/
def get_box_from_info(info, image_id):
    det = []
    data = []
    for anno in info['up']["annotations"]:
        box = anno["head_bbox"]
        score = anno['tracking_id']
        det.append({
            'image_id': image_id,
            'score': score,
            'bbox': box,
            'category_id': 0
        })
        data.append(np.hstack([x1y1wh_x1y1x2y2(box), score, image_id]))
    return det, np.array(data)


def prepare_det(anno):
    data = []
    for det in anno:
        image_id = det['image_id']
        score = det['score']
        box = det['bbox']
        category_id = det['category_id']
        data.append(np.hstack([x1y1wh_x1y1x2y2(box), score, image_id]))
    data = np.array(data)
    ids = data[:, 5].tolist()
    if len(set(ids)) > 1:
        print('more than one image in this list!')
        raise

    return data


def x1y1wh_x1y1x2y2(box):
    x1, y1, w, h = box
    x2, y2 = w + x1 - 1, h + y1 - 1
    return np.array([x1, y1, x2, y2])


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
        info = fisheye(img, ii, visual=visual)
        sv_det, det = get_box_from_info(info, image_id)  # _; det:x1y1wh,score,imageid
        matrial = prepare_xml(det, file_name)
        detections.extend(sv_det)

        profiler.stop('engine')
    np.savez('int8_%d_%d' % m_param, k=detections)
    fisheye.free_Ins()
    del fisheye


def prepare_xml(det, Imgfile):
    xmlfile = Imgfile.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
    if not os.path.exists(xmlfile):
        print('xml file does not exists! %s' % xmlfile)
        raise
    gt_boxes = parse_xml(xmlfile)  # X1Y1x2y2
    image_id = det[:, 5]
    score = det[:, 4]
    det_box = det[:, :4]
    iou = npbbox_iou(gt_boxes[:, :4], det_box)
    gtN = len(gt_boxes)
    ious = np.max(iou, axis=0)
    return np.vstack([ious, score]), gtN


def evalue_map_cal(m_param, img_files):
    print('\n*-* test int8_%d_%d now! *-*\n' % m_param)
    info = np.load('int8_%d_%d.npz' % m_param)
    info.allow_pickle = True
    detections = info['k'].tolist()
    start, end = 0, 0
    if not len(detections): print('No detections!')
    res, TPN = [], 0
    while True:
        start_id = detections[start]['image_id']
        while detections[end]['image_id'] == start_id:
            end += 1
            if end >= len(detections): break
        det = prepare_det(detections[start:end])
        start = end
        Imgfile = img_files[start_id]
        scores, gtn = prepare_xml(det, Imgfile)
        res.append(scores)
        TPN += gtn
        if end >= len(detections): break
    res = np.hstack(res)
    y_score = res[1]
    iou = res[0]
    plt.cla()
    for iou_nms in [0.5, 0.75]:
        print("when IOU threshold is ", iou_nms)
        ytest = iou > iou_nms
        plot_main(ytest.copy(), y_score.copy(), TPN, 'int8_%d_%d.png' % m_param)
    plt.savefig('int8_%d_%d.png' % m_param)


if __name__ == '__main__':
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model = ((640, 640), (640, 1280), (768, 768), (896, 896))
    # files  = ('/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/CH_CLIP_20190816-173844/perimeter_415.jpg',)
    jpg_path = '/home/user/project/dewarp/all_dewarp/perimeter/JPEGImages/'
    json_file = '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/xml.json'
    with open(json_file)as f:
        data = json.load(f)
    coco = COCO(json_file)
    files = []
    img_file = {}
    for id in coco.imgs.keys():
        image = coco.loadImgs(id)[0]['file_name']
        files.append((id, '{}/{}'.format(jpg_path, image)))
        img_file[id] = '{}/{}'.format(jpg_path, image)
    for mm in model:
        # prepare_map_cal(mm, files)
        evalue_map_cal(mm, img_file)
