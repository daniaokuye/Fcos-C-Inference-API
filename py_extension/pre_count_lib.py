import sys
import os
import json
import numpy as np
import copy, time
import cv2, random
import requests
import traceback
from config import cfg_priv, merge_priv_cfg_from_file
from fishEye_lib import FishEye
from colormap import colormap
from ut import find_rect, cross_line, recorder

cfg_file = os.path.join(os.path.split(__file__)[0], 'face_analysis_config.yaml')
merge_priv_cfg_from_file(cfg_file)

N = 100


class FaceCounts(object):
    def __init__(self):
        super().__init__()
        self.FishEye = FishEye()

        self.tracks = dict()
        self.tracks['up'] = dict()
        # 记录输入数组，以便于debug
        self.recorder = recorder(N)
        # 是否画图，可视化来debug
        self.debug = False
        self.II = 0  # 可视化时，保存用的名字自增名称
        self.visual_id = 0  # 可视化是否交叉时，保存用的名字自增名称
        self.curID = 0  # 帧号
        self.set_MACetc = False  # 判断是否有设定mac，进出门店的位置标识线等
        # 发送的基本信息格式send
        self.content = {'media_id': -1, 'media_mac': "", "count_area_id": -1, "count_area_type": -1,
                        "in_num": 0, "out_num": 0, "pass_num": 0, "event_time": 0}

        # self.width = 1920
        # self.in_num = 0
        # self.out_num = 0
        # self.pass_num = 0
        # self.ratio = 0
        # self.out_info = {'list_track': [], 'list_box': [], 'list_id': [], 'solid': [], 'dotted': []}
        # assert cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != "None", \
        #     "cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'] != None"
        # assert len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0, \
        #     "len(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default']) != 0"
        # assert cfg_priv.OTHER.COUNT_MODE == True, \
        #     "cfg_priv.OTHER.COUNT_MODE == True"
        # self.redefine = False
        # self.rect = [[0, 0], [1, 1]]
        # self.entrance_line = [[2, 2], [4, 4]]

    def dummpy(self):
        '''查看是否更新图框；如有更新'''
        pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'set.json')
        if os.path.exists(pth):
            print('set params is true')
            self.set_MACetc = True
            with open(pth, 'r')as f:
                params = json.load(f)
            self.media_id = params['media_id']
            self.media_mac = params['media_mac']
            self.media_rtsp = params['media_rtsp']
            self.record = params['debug']

            self.shopID = params['BUSS.COUNT.ROI_AREA_ID']
            self.lineType = params['BUSS.COUNT.ROI_AREA_TYPE']
            areas = []
            for t, keys in enumerate(['BUSS.COUNT.ROI_SOLID_LINE_AREA', 'BUSS.COUNT.ROI_DOTEED_LINE_AREA']):
                Items = []
                for i, (shopid, Points) in enumerate(params[keys].items()):
                    if shopid not in self.shopID:
                        print('%s should equals to the shop IDS %s:' % (str(shopid), str(self.shopID)))
                        continue
                    # todo: may change order here: rows, cols
                    points = np.array([[int(eval(x) * 2.5), int(eval(y) * 2.5)] for x, y in Points], dtype=np.int32)
                    Items.append(points)
                areas.append(Items)
            self.areas = list(zip(*areas))
            if self.record:
                new_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'D_set.json')
                os.system('mv %s %s' % (pth, new_pth))
            else:
                os.system('rm %s' % pth)

    def canvas(self):
        img = np.ones((self.H, self.W, 3), dtype=np.uint8) * 255
        img[:, :2] = 0
        img[:, -2:] = 0
        img[:2, :] = 0
        img[-2:, :] = 0
        if self.set_MACetc:
            for ii, (solid, dotted) in enumerate(self.areas):
                cur_id = self.shopID[ii]
                for pi in range(len(solid)):
                    cv2.line(img, (solid[(pi + 1) % len(solid)][0], solid[(pi + 1) % len(solid)][1]),
                             (solid[pi][0], solid[pi][1]), (80, 80, 80), 4)
                for pi in range(len(dotted)):
                    cv2.line(img, (dotted[(pi + 1) % len(dotted)][0], dotted[(pi + 1) % len(dotted)][1]),
                             (dotted[pi][0], dotted[pi][1]), (100, 100, 100), 2)
                cv2.putText(img, "shop-%s" % cur_id, (solid[0][0] + 1, solid[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
                cv2.putText(img, "shop-%s" % cur_id, (dotted[0][0] + 1, dotted[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.putText(img, "cur id-%d" % self.curID, (int(self.W * 0.6), int(self.H * 0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (80, 80, 80), 2)
        track = self.out_info['list_track']
        boxes = self.out_info['list_box']
        draw_id = self.out_info['list_id']
        colors = self.out_info['list_color']
        solid = self.out_info['solid']
        dotted = self.out_info['dotted']
        for ii in range(len(track)):
            points = track[ii]
            color = colors[ii]
            cur_id = draw_id[ii]
            line_s, line_d = solid[ii], dotted[ii]
            text = False
            for pi in range(1, len(points)):
                if abs(points[pi][0] - points[(pi - 1) % len(points)][0]) > self.W // 2: continue
                cl = color
                if line_s[pi] != '': cl = (0, 0, 0)
                if line_d[pi] != '': cl = (50, 50, 50)
                cv2.line(img, (points[pi - 1][0], points[pi - 1][1]), (points[pi][0], points[pi][1]), cl, 3)
                text = True
            if text:
                cv2.putText(img, "id: %d" % cur_id, (points[0][0] + 1, points[0][1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        self.II += 1
        cv2.imwrite('vs/base_demo%d.jpg' % self.II, img)

    def visual_check_intersection(self, aax, aay, bbx, bby, ccx, ccy, ddx, ddy):
        minx = min([aax, bbx, ccx, ddx]) - 10
        miny = min([aay, bby, ccy, ddy]) - 10
        maxx = max([aax, bbx, ccx, ddx]) - 10
        maxy = max([aay, bby, ccy, ddy]) - 10
        img = np.ones((maxy - miny + 30, maxx - minx + 30, 3), dtype=np.uint8) * 255
        img[:, :2] = 0
        img[:, -2:] = 0
        img[:2, :] = 0
        img[-2:, :] = 0
        cv2.line(img, (int(aax - minx), int(aay - miny)), (int(bbx - minx), int(bby - miny)), (0, 0, 0), 1)
        cv2.line(img, (int(ccx - minx), int(ccy - miny)), (int(ddx - minx), int(ddy - miny)), (0, 0, 0), 1)
        self.visual_id += 1
        cv2.imwrite('vs/check%d.jpg' % self.visual_id, img)

    def deter_in_out(self, aax, aay, bbx, bby):
        status = []
        if not self.set_MACetc:
            return status
        for i, (solid, dotted) in enumerate(self.areas):
            shop = self.shopID[i]
            for ii in range(len(solid)):
                ccx, ccy = solid[ii]
                ddx, ddy = solid[(ii + 1) % len(solid)]
                cross = cross_line(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                if cross:
                    if self.debug: self.visual_check_intersection(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                    status.append(['s', shop])

            for ii in range(len(dotted)):
                ccx, ccy = dotted[ii]
                ddx, ddy = dotted[(ii + 1) % len(dotted)]
                cross = cross_line(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                if cross:
                    if self.debug: self.visual_check_intersection(aax, aay, bbx, bby, ccx, ccy, ddx, ddy)
                    status.append(['d', shop])
        return status

    def get_tracks(self, img_data, current_id, max_lost_frames=5):
        if self.set_MACetc:
            self.statics_in = dict.fromkeys(self.shopID, 0)
            self.statics_out = dict.fromkeys(self.shopID, 0)
            self.statics_passby = dict.fromkeys(self.shopID, 0)
        up_and_down_frames = 2
        for person in img_data['up']["annotations"]:
            track_id = person["tracking_id"]
            global_id = person["global_id"]
            box_xywh = person["head_bbox"]
            position = [int(box_xywh[0] + box_xywh[2] / 2), int(box_xywh[1] + box_xywh[3] / 2)]

            if track_id in self.tracks['up']:
                self.tracks['up'][track_id]['boxes'] = box_xywh
                self.tracks['up'][track_id]['track'].append(position)
                self.tracks['up'][track_id]['latest_frame'] = current_id
                self.tracks['up'][track_id]['solid'].append('')
                self.tracks['up'][track_id]['dotted'].append('')
            else:
                self.tracks['up'][track_id] = dict()
                self.tracks['up'][track_id]['boxes'] = box_xywh
                self.tracks['up'][track_id]['track'] = []
                self.tracks['up'][track_id]['track'].append(position)
                self.tracks['up'][track_id]['draw_id'] = global_id
                self.tracks['up'][track_id]['status'] = True
                self.tracks['up'][track_id]['draw'] = True
                self.tracks['up'][track_id]['start_frame'] = current_id
                self.tracks['up'][track_id]['latest_frame'] = current_id
                self.tracks['up'][track_id]['solid'] = ['']
                self.tracks['up'][track_id]['dotted'] = ['']
            if len(self.tracks['up'][track_id]['track']) > 1:
                aax, aay = self.tracks['up'][track_id]['track'][-2]
                bbx, bby = position
                # todo: cross line
                rets = [] if abs(aax - bbx) > self.W // 2 else self.deter_in_out(aax, aay, bbx, bby)
                for ret in rets:
                    types, shop = ret
                    id_record = len(self.tracks['up'][track_id]['solid'])
                    old_key_S, old_key_D, new_key_S, new_key_D = (-1,) * 4
                    idx_key_s = '_'.join([str(shop), 's'])
                    idx_key_d = '_'.join([str(shop), 'd'])

                    if idx_key_s in self.tracks['up'][track_id]:
                        old_key_S = self.tracks['up'][track_id][idx_key_s]
                    if idx_key_d in self.tracks['up'][track_id]:
                        old_key_D = self.tracks['up'][track_id][idx_key_d]
                    '''计数逻辑：
                    首先，判断是否碰线，碰的什么样的线；
                    其次，分析碰线次序，来判断是出还是进。
                        口口型1.碰一次实线后，至少碰一次虚线后：进；反之，出；碰两次实线，一段时间没有记录或一段时间没碰虚线，路过
                        回字型2.碰一次实线后，碰一次虚线：进；反之，出；碰两次实线，路过。
                        口字型3.将碰线视为：只有路过；或只有进出。
                    最后，对于情况1，需对疑是路过进行标记，最终记做路过或进店（店口徘徊）
                    TODO：未来，可添加位移量。目前只考虑碰线这一事件变量。
                    所以，目前使用单个整数数字来表示碰线次序，浮点数来标记。单个数字即可完成。
                    以上，为value值的意义和表示方法。
                    key包含有shop信息，线类型信息。
                    故目前使用key-value对来表示关系，key-value属于不同的id字典。
                    '''
                    # 如果经过一段时间仍没有消掉mark，或者有更新记录，那么就计入passby；删除实线框记录
                    if isinstance(old_key_D, float) and (abs(id_record - old_key_D) < 20 or types != ''):
                        self.statics_passby[shop] += 1
                        self.tracks['up'][track_id][idx_key_d] = int(self.tracks['up'][track_id][idx_key_d])
                        self.tracks['up'][track_id][idx_key_s] = -1

                    if types == 's':
                        self.tracks['up'][track_id]['solid'][-1] = shop
                        if id_record - old_key_S < up_and_down_frames and id_record >= 2: old_key_S = id_record
                        self.tracks['up'][track_id][idx_key_s] = id_record
                        new_key_S = id_record
                    if types == 'd':
                        self.tracks['up'][track_id]['dotted'][-1] = shop
                        if id_record - old_key_D < up_and_down_frames and id_record >= 2: old_key_D = id_record
                        self.tracks['up'][track_id][idx_key_d] = id_record
                        new_key_D = id_record

                    # 让new始终作为最新记录
                    if new_key_S == -1:
                        old_key_S, new_key_S = new_key_S, old_key_S
                    if new_key_D == -1:
                        old_key_D, new_key_D = new_key_D, old_key_D
                    # 用记录来计数
                    door_status = self._count(old_key_S, old_key_D, new_key_S, new_key_D, shop, self.lineType[shop])
                    # 成功计数一次之后，就需要更新记录了
                    if door_status == 'passby':  # 留意虚线框
                        if idx_key_d not in self.tracks['up'][track_id]:
                            self.tracks['up'][track_id][idx_key_d] = -0.5
                        else:
                            self.tracks['up'][track_id][idx_key_d] += 0.5
                    if door_status == 'in':  # 删除实线框记录
                        self.tracks['up'][track_id][idx_key_s] = -1
                    if door_status == 'out':  # 删除虚线框记录
                        v = -0.5 if isinstance(old_key_D, float) else -1
                        self.tracks['up'][track_id][idx_key_d] = v

        for track_id in self.tracks['up'].keys():
            if current_id - self.tracks['up'][track_id]['latest_frame'] > max_lost_frames:
                img_data['up']["delete_tracking_id"].append(track_id)
            if track_id in img_data['up']["delete_tracking_id"]:
                # if track_id == 4:
                #     print()
                self.tracks['up'][track_id]['status'] = False

    def smart_judge(self, key, track_id):
        '''拼接的基本逻辑是断点的判断；
        但是，目前使用的计数逻辑不是可逆向追溯，所以断点重连，对计数没有帮助。
        但是可以想办法，利用区间计算是否构成进出店等；但有多次计数的风险；
        也可以在删除本计数点时，断言进出店情况。
        '''
        # can a stop point of a line being a start point of another line?
        # 拼接断点相近，且位置相近的实例
        dead_id = self.tracks[key][track_id]
        up_and_down_frames = 2

        for cur_key, instance in self.tracks[key].items():
            if cur_key == track_id: continue
            if 0 <= instance['start_frame'] - dead_id['latest_frame'] < 2 * up_and_down_frames:
                if not (len(instance['track']) or len(dead_id['track'])): continue
                dis = np.array(instance['track'][0]) - np.array(dead_id['track'][-1])
                if np.sum(np.abs(dis) < 10) == 2:
                    # connect them
                    dead_id['track'].extend(instance['track'])
                    dead_id['solid'].extend(instance['solid'])
                    dead_id['dotted'].extend(instance['dotted'])
                    instance['track'] = dead_id['track']
                    instance['solid'] = dead_id['solid']
                    instance['dotted'] = dead_id['dotted']
                    instance['start_frame'] = dead_id['start_frame']
                    self.tracks[key].pop(track_id)
                    return True
        return False

    def count_num(self):
        tracks_tmp_up = copy.deepcopy(self.tracks['up'])
        track_ids = tracks_tmp_up.keys()
        pop_ids = []
        for track_id in track_ids:
            occur = (self.curID == self.tracks['up'][track_id]['latest_frame'])
            if self.tracks['up'][track_id]['status']:
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks['up'][track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks['up'][track_id])
            else:
                is_combine = self.smart_judge('up', track_id)
                if is_combine: continue
                if cfg_priv.OTHER.COUNT_DRAW:
                    if len(self.tracks['up'][track_id]['track']) > 0 and occur:
                        self.draw_track(self.tracks['up'][track_id])
                pop_ids.append(track_id)
        for track_id in pop_ids:
            self.tracks['up'].pop(track_id)

    # 计数逻辑的核心代码
    def _count(self, old_key_S, old_key_D, new_key_S, new_key_D, shop, lineType):
        '''注意：passby比较麻烦，因为经过两次实现框，且短时间内不经过虚线
        注意2：一旦计数成功，需要清空旧数据，以防虚假判断'''
        double_solid = False  # 两次碰触实线
        if old_key_S > 0 and new_key_S > 0 and old_key_D <= 0 and new_key_D <= 0:
            double_solid = True
        if lineType == 1:  # 可以不用考虑经过两次实线框，或两次虚线框。why，经过了实线框，又经过虚线框，肯定是经过了进门线
            if double_solid:
                return 'passby'  # mark and fcous on dot line
            if new_key_S > 0 and new_key_D > 0 and new_key_D >= new_key_S:
                self.statics_in[shop] += 1
                return 'in'
            if new_key_D > 0 and new_key_S > 0 and new_key_D < new_key_S:
                self.statics_out[shop] += 1
                return 'out'
        elif lineType == 2:
            if new_key_D > 0 and new_key_S > 0 and new_key_D >= new_key_S:
                self.statics_in[shop] += 1
                return 'in'
            if double_solid or (new_key_D > 0 and new_key_S > 0 and new_key_D < new_key_S):
                self.statics_out[shop] += 1
                return 'out'
        else:  # lineType == 3:
            if double_solid:
                self.statics_out[shop] += 1
                return 'out'
            elif new_key_S > 0:
                self.statics_in[shop] += 1
                return 'in'

    # def draw_scope(self, rec, entrance_line):
    #     self.out_info['rec'] = rec
    #     self.out_info['entrance_line'] = entrance_line
    #
    # def draw_num(self):
    #     self.out_info['pass_by'] = self.pass_num
    #     self.out_info['entran'] = self.in_num
    #     self.out_info['ratio'] = round(self.ratio, 2)

    def draw_track(self, track):
        color_map = colormap()
        color = int(color_map[track['draw_id'] % 79][0]), int(color_map[track['draw_id'] % 79][1]), int(
            color_map[track['draw_id'] % 79][2])

        if track['draw']:
            if cfg_priv.OTHER.COUNT_DRAW_LESS:
                new_track = track['track'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
                solid = track['solid'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
                dotted = track['dotted'][-cfg_priv.OTHER.DRAW_TRACK_NUM:]
            else:
                new_track = track['track']
                solid = track['solid']
                dotted = track['dotted']
            self.out_info['solid'].append(solid)
            self.out_info['dotted'].append(dotted)
            self.out_info['list_track'].append(new_track)
            self.out_info['list_box'].append(track['boxes'])
            self.out_info['list_id'].append(track['draw_id'])
            self.out_info['list_color'].append(color)

    def send(self):
        def oneItem(content):
            if self.debug:
                if content["in_num"] or content['out_num'] or content['pass_num']:
                    self.canvas()
                    print(content)
            else:
                requests.post(url='http://172.16.104.247:5000/flow/pvcount', data=content)

        in_num, out_num, pass_num = 0, 0, 0

        if not self.set_MACetc:
            content = self.content
            if content["in_num"] or content['out_num'] or content['pass_num']:
                oneItem(content)
        else:
            for shop in self.shopID:
                content = {'media_id': self.media_id, 'media_mac': self.media_mac, "count_area_id": shop,
                           "count_area_type": self.lineType[shop], "in_num": self.statics_in[shop],
                           "out_num": self.statics_out[shop], "pass_num": self.statics_passby[shop],
                           "event_time": int(time.time())}
                in_num += content["in_num"]
                out_num += content['out_num']
                pass_num += content['pass_num']
                if content["in_num"] or content['out_num'] or content['pass_num']:
                    oneItem(content)
        # self.out_info.update({'entran': in_num, 'pass_by': pass_num, 'out_num': out_num})

    # self.pad = 0
    # def get_line_offset(self, v_line, v_pad):
    #     return [[v_line[0][0] + v_pad, v_line[0][1]], [v_line[1][0] + v_pad, v_line[1][1]]]
    # def clear_all(self):
    #     self.in_num = 0
    #     self.out_num = 0
    #     self.pass_num = 0
    #     self.ratio = 0
    # if self.redefine:
    #     self.redefine = False
    #     if cfg_priv.BUSS.COUNT.ROI_AREA['default'] == "None" or cfg_priv.BUSS.COUNT.ROI_AREA[
    #         'default'] is None or len(cfg_priv.BUSS.COUNT.ROI_AREA['default']) == 0:
    #         rect = ([0 + self.pad, 0], [W + self.pad, H // 2])
    #     else:
    #         rect = find_rect(cfg_priv.BUSS.COUNT.ROI_AREA['default'], (H, W), self.pad)
    #     entrance_line = self.get_line_offset(cfg_priv.BUSS.COUNT.ENTRANCE_LINE['default'], self.pad)
    # else:
    #     rect, entrance_line = self.rect, self.entrance_line

    def __call__(self, scores, boxes, classes, ratio_h, ratio_w, H, W):
        self.H, self.W = H, W
        self.curID += 1
        # 除跟踪线、人体框、人体id以及相对应的颜色外，额外的数据：
        # 进、出、路过
        self.out_info = {'list_track': [], 'list_box': [], 'list_id': [], 'list_color': [],
                         'entran': 0, 'pass_by': 0, 'out_num': 0, 'solid': [], 'dotted': [],
                         'rec': [[0, 0], [1, 1]], 'entrance_line': [[2, 2], [4, 4]]}

        C_scores, C_boxes, C_classes = scores.copy(), boxes.copy(), classes.copy()
        status = False
        try:
            ##################################
            self.dummpy()
            single_img_info_dict = self.FishEye(scores, boxes, classes, ratio_h, ratio_w, H, W)
            self.get_tracks(single_img_info_dict, self.curID)
            if len(single_img_info_dict['up']['annotations']): status = True
            self.count_num()
            # if cfg_priv.OTHER.COUNT_DRAW:
            #     self.draw_num()
            self.send()
            ##################################
        except Exception as e:
            traceback.print_exc(e)
        if self.set_MACetc and self.record and status:
            self.recorder.save(C_scores, C_boxes, C_classes)
        return self.out_info


if __name__ == '__main__':
    from runProject import test_set

    test_set(False)
    fc = FaceCounts()
    fc.debug = True
    npz = '../build/xxx_%d.npz'
    n = N = 100
    while True:
        if not os.path.exists(npz % n) or n / 100 >= 20: break
        print(npz % n)
        data = np.load(npz % n)
        n += N
        data.allow_pickle = True
        scores, classes, boxes = data['s'], data['c'], data['b']
        print(fc.set_MACetc)
        for s, c, b in zip(scores, classes, boxes):
            res = fc(s, b, c, 1.5, 1.6, 1920, 2880)
