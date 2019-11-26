# url: https://blog.csdn.net/summermaoz/article/details/70178190
from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np

__all__ = ['read_all', 'parse_xml']


def read_all():
    basePath = '/home/user/project/dewarp/all_dewarp/'
    for mode in ['perimeter', 'center']:
        base = os.path.join([basePath, mode, 'Annotations'])
        for dir in os.listdir(base):
            AnnoPath = os.path.join([base, dir])
            ImgPath = os.path.join([basePath, mode, 'JPEGImages', dir])
            imagelist = os.listdir(ImgPath)
            for image in imagelist:
                print('a new image:', image)
                image_pre, ext = os.path.splitext(image)
                imgfile = ImgPath + image
                xmlfile = AnnoPath + image_pre + '.xml'
                yield imgfile, xmlfile


def parse_xml(xmlfile):
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')

    i = 1
    boxes = []
    for objects in objectlist:
        # print objects
        namelist = objects.getElementsByTagName('name')
        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        # print(objectname)
        if (objectname != 'person'): continue

        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            try:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                w = x2 - x1
                h = y2 - y1
                boxes.append([x1, y1, x2, y2, w, h])
            except Exception as e:
                print(e)
    return np.array(boxes)


def expand_crop(imgfile, box):
    x1, y1, x2, y2, w, h = box
    img = Image.open(imgfile)
    width, height = img.size

    obj = np.array([x1, y1, x2, y2])
    shift = np.array([[0.8, 0.8, 1.2, 1.2], [0.9, 0.9, 1.1, 1.1], [1, 1, 1, 1],
                      [0.8, 0.8, 1, 1], [1, 1, 1.2, 1.2],
                      [0.8, 1, 1, 1.2], [1, 0.8, 1.2, 1],
                      [(x1 + w * 1 / 6) / x1, (y1 + h * 1 / 6) / y1,
                       (x2 + w * 1 / 6) / x2, (y2 + h * 1 / 6) / y2],
                      [(x1 - w * 1 / 6) / x1, (y1 - h * 1 / 6) / y1,
                       (x2 - w * 1 / 6) / x2, (y2 - h * 1 / 6) / y2]])

    XYmatrix = np.tile(obj, (9, 1))
    cropboxes = XYmatrix * shift

    for cropbox in cropboxes:
        # print 'cropbox:',cropbox
        minX = max(0, cropbox[0])
        minY = max(0, cropbox[1])
        maxX = min(cropbox[2], width)
        maxY = min(cropbox[3], height)

        cropbox = (minX, minY, maxX, maxY)
        cropedimg = img.crop(cropbox)
        # cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.jpg')
        # i += 1
