#!/usr/bin/python

# url: https://www.cnblogs.com/marsggbo/p/11152462.html

import sys
import os
import json
import random, cv2
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES =   {'person': 1, 'head': 2}

# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
imgids = {}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    ids = random.randint(0, 10000)
    while ids in imgids:
        ids = random.randint(0, 10000)
    imgids[ids] = None
    return ids


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        # path = get(root, 'path')
        filename = line.replace('xml', 'jpg')
        if not os.path.exists(os.path.join(xml_dir.replace('Annotations', 'JPEGImages'), filename)):
            raise NotImplementedError('%s not found' % (filename))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category != 'person': continue
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    for key in ["images", "annotations", "categories"]:
        print(key, len(json_dict[key]))
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    print(categories)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


def split_box(xmin, ymin, xmax, ymax, tmpids, category_id, bnd_id, ratio_h, ratio_w,
              pad=128, H=960, W=1920, least_w=0.2):
    anns = []
    o_height = ymax - ymin
    add_value = 0
    # 1
    if xmin < (W + pad) and ymax <= H:
        o_width = min(W + pad - xmin, xmax - xmin)
        if o_width / (xmax - xmin) > least_w:
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': tmpids[0], 'bbox':
                list(map(int, [xmin * ratio_w, ymin * ratio_h, o_width * ratio_w, o_height * ratio_h])),
                   'category_id': category_id, 'id': bnd_id + add_value,
                   'ignore': 0, 'segmentation': []}
            anns.append(ann)
            add_value += 1
    # 2
    if xmax > W and ymax <= H:
        new_xmin = max(xmin, W)
        o_width = xmax - new_xmin
        if o_width / (xmax - xmin) > least_w:
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': tmpids[1], 'bbox':
                list(map(int, [(new_xmin - W) * ratio_w, ymin * ratio_h, o_width * ratio_w, o_height * ratio_h])),
                   'category_id': category_id, 'id': bnd_id + add_value,
                   'ignore': 0, 'segmentation': []}
            anns.append(ann)
            add_value += 1
    if xmin < (W // 2 + pad) and ymin > H:
        o_width = min(W // 2 + pad - xmin, xmax - xmin)
        if o_width / (xmax - xmin) > least_w:
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': tmpids[1], 'bbox':
                list(map(int, [(xmin + W // 2) * ratio_w, (ymin - H) * ratio_h,
                               o_width * ratio_w, o_height * ratio_h])),
                   'category_id': category_id, 'id': bnd_id + add_value,
                   'ignore': 0, 'segmentation': []}
            anns.append(ann)
            add_value += 1
    # 3
    if xmax > (W // 2) and ymin > H:
        new_xmin = max(xmin, W // 2)
        o_width = xmax - new_xmin
        if o_width / (xmax - xmin) > least_w:
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': tmpids[2], 'bbox':
                list(map(int, [(new_xmin - W // 2) * ratio_w, (ymin - H) * ratio_h,
                               o_width * ratio_w, o_height * ratio_h])),
                   'category_id': category_id, 'id': bnd_id + add_value,
                   'ignore': 0, 'segmentation': []}
            anns.append(ann)
            add_value += 1
    if xmax < pad and ymax <= H:
        o_width = min(pad - xmin, xmax - xmin)
        if o_width / (xmax - xmin) > least_w:
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': tmpids[2], 'bbox':
                list(map(int, [(xmin + W) * ratio_w, ymin * ratio_h,
                               o_width * ratio_w, o_height * ratio_h])),
                   'category_id': category_id, 'id': bnd_id + add_value,
                   'ignore': 0, 'segmentation': []}
            anns.append(ann)
            add_value += 1

    return anns, add_value


def reorderAnno(anns, tmpids):
    image_ID, ids = {}, []
    for a in anns:
        ids.append(a['id'])
        ii = a['image_id']
        if ii not in image_ID: image_ID[ii] = []
        image_ID[ii].append(a)
    ids.sort()
    i = 0
    for k in tmpids:
        if k not in image_ID: continue
        for a in image_ID[k]:
            a['id'] = ids[i]
            i += 1
    return image_ID


def convertSplitAnno(xml_list, xml_dir, json_file, Mh=960, Mw=1920, show=False):
    list_fp = open(xml_list, 'r')
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s" % (line), end=', ')
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        # path = get(root, 'path')
        filename = line.replace('xml', 'jpg')
        if not os.path.exists(os.path.join(xml_dir.replace('Annotations', 'JPEGImages'), filename)):
            raise NotImplementedError('%s not found' % (filename))
        ## The filename must be a number
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        pad = int(128 * 1.0)
        ratio_h = Mh / (height // 2)
        ratio_w = Mw / (width // 3 * 2 + pad)
        # print(ratio_h, ratio_w)
        fn = os.path.splitext(filename)[0]
        tmpids = []
        for i in range(3):
            x_fn = fn + '_%d.jpg' % (i + 1)
            if not os.path.exists(os.path.join(xml_dir.replace('Annotations', 'JPEGImages'), x_fn)):
                raise NotImplementedError('%s not found' % (x_fn))
            image_id = get_filename_as_int(x_fn)
            tmpids.append(image_id)
            image = {'file_name': x_fn, 'height': Mh, 'width': Mw, 'id': image_id}
            json_dict['images'].append(image)
        ## Currently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        tmp = []
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category == 'face': continue
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            ann, add_value = split_box(xmin, ymin, xmax, ymax, tmpids, category_id, bnd_id,
                                       ratio_h, ratio_w, pad, height // 2, width // 3 * 2)
            tmp.extend(ann)
            bnd_id = bnd_id + add_value
        tmpNew = reorderAnno(tmp, tmpids)
        for k in tmpids:
            if k not in tmpNew: continue
            json_dict['annotations'].extend(tmpNew[k])
        if show:
            for k in tmpids:
                if k not in tmpNew: continue
                annlist = tmpNew[k]
                imageid = annlist[0]['image_id']
                xxfn = None
                for x in json_dict['images']:
                    if x['id'] == imageid:
                        xxfn = x['file_name']
                        break
                if xxfn == None:
                    raise AttributeError('No id in it')
                jpgfile = os.path.join(xml_dir.replace('Annotations', 'JPEGImages'), xxfn)
                outfile = os.path.join(os.getcwd(), xxfn)
                img = cv2.imread(jpgfile)
                for aa in annlist:
                    box = aa['bbox']
                    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  (255, 255, 255), thickness=1)
                base_out = os.path.dirname(outfile)
                if not os.path.exists(base_out): os.mkdir(base_out)
                cv2.imwrite(outfile, img)
        if len(tmpNew) != 3:
            print(len(tmpNew), '*' * 10, len(tmp))
        else:
            print(len(tmpNew), len(tmp))

    for key in ["images", "annotations", "categories"]:
        print(key, len(json_dict[key]))
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    print(categories, json_file)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('3 auguments are need.')
        print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json' % (sys.argv[0]))
        exit(1)
    # convert(sys.argv[1], sys.argv[2], sys.argv[3])
    convertSplitAnno(sys.argv[1], sys.argv[2], sys.argv[3])

    # convertSplitAnno('/home/user/project/dewarp/all_dewarp/perimeter/Annotations/xml.txt',
    #                  '/home/user/project/dewarp/all_dewarp/perimeter/Annotations',
    #                  '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/xml.json')
