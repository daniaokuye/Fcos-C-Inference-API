import os
import time
import ctypes, cv2
import numpy as np
from utilty import get_blob


def get_lib(engine_file):
    kernel = '/home/user/project/retinanet/torch_extension/build/libinfer_2.so'
    print('kernel:', kernel)
    lib = ctypes.cdll.LoadLibrary(kernel)
    info = init(lib, engine_file)
    print(info)
    return lib, info


def init(lib, engine_file, name=b'engine16'):
    # engine_file = b"/home/user/weight/engine16.plan"
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
    # lib.run(dataptr_in, scoreptr_in, boxptr_in, clsptr_in, batch, name,
    #         engine_file, _array)

    return score, box, cls


def old_shedule():
    engine_file = b"/home/user/weight/engine8_640_1280.plan"
    lib, info = get_lib(engine_file)
    img = cv2.imread('./build/1001.jpg').astype(np.float32)
    a = img[:640] - 102.9801
    b = img[640:640 * 2] - 115.9465
    c = img[640 * 2:] - 122.7717
    img = np.stack([a, b, c])
    print(img.shape, img.dtype)
    img = img.transpose(0, 3, 1, 2)
    data = img.copy()
    print(data.shape, data.dtype)
    scores, boxes, classes = test(lib, data, batch=3)
    idx = (scores >= 0.5) * (classes == 0)
    print('there are %d suit condition' % (idx.sum()), '\n', boxes[idx])


def test_simple_infer(m_param):
    name = b"/home/user/project/run_retina/weights/fcos_int8_A_%d_%d.plan" % m_param
    caliFile = "/home/user/project/run_retina/weights/calibration_files.txt"
    kernel = '/home/user/project/run_retina/build/libinfer_test.so'
    print('kernel:', kernel, name)
    lib = ctypes.cdll.LoadLibrary(kernel)
    # a = np.load('/home/user/weight/xxx.npz')
    # if not a.allow_pickle: a.allow_pickle = True
    # x = a['x']
    # with open(caliFile)as f:
    #     jpg = f.readline().strip()
    jpg = '/home1/datasets/coco/images/val2017/000000414170.jpg'
    img = cv2.imread(jpg)
    img = np.hstack([img, img, img])
    img, ratio = get_blob(img, *m_param)
    img = img.astype(np.float32)
    # img = img - np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    # img = img / np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    img = img - np.array([103.52, 116.28, 123.675], dtype=np.float32)
    img = img / np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img = img[np.newaxis]
    x = img.transpose(0, 3, 1, 2)

    print(x.shape, x.dtype)
    outbox = np.zeros(4 * 100, dtype=x.dtype)
    dataptr_in = x.ctypes.data_as(ctypes.c_char_p)
    boxptr_in = outbox.ctypes.data_as(ctypes.c_char_p)
    lib.fun(dataptr_in, m_param[1], m_param[0], name, boxptr_in)
    # outbox = outbox.reshape(-1, 4)
    # print(outbox * np.array([1.0648, 1.065, 1.0648, 1.065]))
    # for b in outbox:
    #     cv2.rectangle(image, 左下角坐标, 右上角坐标, color, 线条粗度)


def get_calibration_files(calibration_images, batch=8, calibration_batches=10):
    if not os.path.isdir(calibration_images):
        print(calibration_images, ' is not exists!')
        return
    calibration_files = []
    import glob, random
    file_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for ex in file_extensions:
        calibration_files += glob.glob("{}/*{}".format(calibration_images, ex), recursive=True)
    # Only need enough images for specified num of calibration batches
    if len(calibration_files) >= calibration_batches * batch:
        calibration_files = calibration_files[:(calibration_batches * batch)]
    else:
        print('Only found enough images for {} batches. Continuing anyway...'.format(
            len(calibration_files) // batch))

    random.shuffle(calibration_files)
    with open('calibration_files.txt', 'w')as f:
        for file in calibration_files:
            f.writelines(file + '\n')


if __name__ == '__main__':
    # get_calibration_files('/home1/datasets/coco/images/val2017')
    test_simple_infer((896, 1792))
