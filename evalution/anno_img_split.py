import os
import time
import ctypes, cv2
import numpy as np


def get_fisheye_data_batch2(filename, Mh=960, Mw=1920):
    s1 = time.time()
    data = cv2.imread(filename)
    h, w, c = data.shape
    pad = int(128 * 1.0)  # 2.5
    W = int(2 * w // 3)  # 960
    ratio = ((W + pad) / Mw, h // 2 / Mh)
    img = np.concatenate((data[:h // 2], data[h // 2:], data[:h // 2, : pad]),
                         axis=1)  # (  960, 5760+pad, 3)
    img1 = img[:, :W + pad]
    img2 = img[:, W:2 * W + pad]
    img3 = img[:, W * 2:]
    img1 = cv2.resize(img1, (Mw, Mh))
    img2 = cv2.resize(img2, (Mw, Mh))
    img3 = cv2.resize(img3, (Mw, Mh))
    fn = os.path.splitext(filename)[0]
    cv2.imwrite(fn + '_1.jpg', img1)
    cv2.imwrite(fn + '_2.jpg', img2)
    cv2.imwrite(fn + '_3.jpg', img3)
    s4 = time.time()
    # print("split: %.3fs " % (s4 - s1))


if __name__ == '__main__':
    # xmlfile = os.path.join(os.getcwd(), 'Annotations/xml.txt')
    root = '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/'
    xmlfile = '/home/user/project/dewarp/all_dewarp/perimeter/Annotations/xml.txt'
    with open(xmlfile)as f:
        for line in f.readlines():
            filename = line.replace('xml', 'jpg').strip()
            jpgfile = os.path.join(root.replace('Annotations', 'JPEGImages'), filename)
            if not os.path.exists(jpgfile):
                raise NotImplementedError('%s not found' % (jpgfile))
            get_fisheye_data_batch2(jpgfile)
