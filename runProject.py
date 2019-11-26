import cv2
import os, sys
import subprocess


class Fisheye():
    def __init__(self):
        self.pwd = os.path.dirname(os.path.realpath(__file__)) + '/build'

    def build_yaml(self, media_id, status='n', baseMap='n'):
        param = "%YAML:1.0\nstop:" + " %s\npic: %s\n" % (status, baseMap)
        self.yaml = self.pwd + '/param_%d.yaml' % media_id
        with open(self.yaml, 'w')as f:
            f.writelines(param)
        os.system('cat %s' % self.yaml)
        # print(self.yaml)

    def start(self, media_id, media_mac, media_rtsp):
        print("start")
        print(media_id, media_mac, media_rtsp)
        self.build_yaml(media_id)
        os.system("echo $DISPLAY")
        self.runlog = 'run_%d.log' % media_id
        cmd = self.pwd + "/infer_ --ipt %s --opt a --media_id %d --mac %s --yaml %s >run_%d.log 2>&1" % \
              (media_rtsp, media_id, media_mac, self.yaml, media_id)
        print(cmd)
        subprocess.call(cmd, shell=True)
        return True

    def stop(self, media_id, **kwargs):
        print("stop")
        self.build_yaml(media_id, status='y')
        self._del()
        return True

    def set(self, media_id, set_type, params):
        print("set")
        print(media_id, set_type, params)
        return True

    def getpic(self, media_id, **kwargs):
        print("get Base map")
        self.build_yaml(media_id, baseMap='y')
        pic = self.pwd + '/param_%d.jpg' % media_id
        print(pic)
        return True, pic

    def _del(self):
        print('delete temp files')
        os.system("rm %s" % self.yaml)
        os.system("rm %s" % self.runlog)


if __name__ == '__main__':
    fe = Fisheye()
    status = sys.argv[1]
    md = int(sys.argv[2])
    # print('md:', md, 'status:', status)
    # print(status)
    if status == 's':  # start
        # fe.start(md, "00-02-D1-83-83-71", 'rtsp://root:admin123@172.16.105.86:554/live.sdp')
        fe.start(md, "00-02-D1-83-83-71", './build/s.mp4')
    elif status == 'e':  # end
        fe.stop(md)
    else:
        fe.getpic(md)
