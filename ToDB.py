# encoding: utf-8
import base64
import json
import time
from PIL import Image
import numpy as np
import io
import cv2, os
import paho.mqtt.client as mqtt


# ffmpeg 推流：https://blog.csdn.net/Mind_programmonkey/article/details/102732555
class connectDB():
    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect("127.0.0.1", 1883, 60)
        # self.client.loop_forever()

    def push_out(self, media_id, media_mac, image_id, image):
        value0 = {}
        print(image_id)
        value0['media_id'] = media_id  # (media_id % 500)
        value0['media_mac'] = media_mac
        # print(1, '+' * 10)
        img = Image.fromarray(image[..., ::-1].copy())
        pth = os.path.join(os.getcwd(), 'test_%d.jpg' % (image_id % 10))
        # print('=============PIL save path: ', pth)
        img.save(pth)
        with open(pth, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        value0['picfile'] = encoded_image
        value0['image_id'] = image_id
        value0['format'] = "image/jpeg"
        param0 = json.dumps(value0)
        self.client.publish(media_mac, param0, 0)


def main(mac1):
    fishObj = connectDB()
    i = 0
    while 1:
        i = i + 1
        img = cv2.imread('a.jpg')
        img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        fishObj.push_out(0, mac1, i, img)
    fishObj.client.loop_forever()


if __name__ == '__main__':
    # 测试的mac。(mq topic 默认使用mac 地址)
    main("00-02-D1-83-83-71")
