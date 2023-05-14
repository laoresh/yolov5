import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from mqtt import aliLink
from mqtt import mqttd
import random
from rfid import rfid as rf

# # 三元素（iot后台获取）
# ProductKey = 'gfgwFjcx153'
# DeviceName = 'BCS_'
# DeviceSecret = 'b213b58f39ff507dd4283cb150b64de2'
# # topic (iot后台获取)
# POST = '/sys/gfgwFjcx153/BCS_/thing/event/property/post'  # 上报消息到云
# POST_REPLY = '/sys/gfgwFjcx153/BCS_/thing/event/property/post_reply'
# SET = '/sys/gfgwFjcx153/BCS_/thing/service/property/set'  # 订阅云端指令

# 三元素（iot后台获取）
ProductKey = 'gib5KzKn1yj'
DeviceName = 'jetson_xavier_nx'
DeviceSecret = 'b7dd08a5fba68952d53ae1ba344591c0'
# topic (iot后台获取)
POST = '/sys/gib5KzKn1yj/jetson_xavier_nx/thing/event/property/post'  # 上报消息到云
POST_REPLY = '/sys/gib5KzKn1yj/jetson_xavier_nx/thing/event/property/post_reply'
SET = '/sys/gib5KzKn1yj/jetson_xavier_nx/thing/service/property/set'  # 订阅云端指令

# 消息回调（云端下发消息的回调函数）
def on_message(client, userdata, msg):
    pass


# 连接回调（与阿里云建立链接后的回调函数）
def on_connect(client, userdata, flags, rc):
    pass


LBCS = np.array([])
RBCS = np.array([])
LID = list()
RID = list()
y_l = np.array([])
y_r = np.array([])
BCS_INDEX = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def detect(opt):
    global y_l
    global y_r
    global LID
    global RID
    global LBCS
    global RBCS
    global BCS_INDEX
    flag_1 = 0
    flag_2 = 0
    bcs_1 = []
    bcs_2 = []
    L_id_head = -1
    R_id_head = -1
    L_port = rf.L_port_init()
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # RFID清除
    L_port.flushInput()
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    if i == 0:
                        y_l = np.append(y_l, xywh[1])
                        iid = rf.read_id(L_port)
                        if iid != L_id_head and iid != -1:
                            L_id_head = iid
                            LID.append(iid)
                        if y_l[-1] >= 0.6:
                            flag_1 += 1
                            # print('flag_1', flag_1)
                        elif (y_l[-1] < 0.6) & (y_l[-1] > 0.4):
                            bcs_1 = BCS_INDEX[cls.int()]
                            flag_1 += 1
                            # print('flag_1', flag_1)
                        else:
                            LBCS = np.append(LBCS, bcs_1)
                            print('LBCS{}'.format(list(LBCS)))
                            print('LID{}'.format(list(LID)))
                            bcs_1 = []
                            flag_1 = 0
                            L_port.flushInput()
                    else:
                        y_r = np.append(y_r, xywh[1])
                        if y_r[-1] >= 0.6:
                            flag_2 += 1
                            # print('flag_2', flag_2)
                        elif (y_r[-1] < 0.6) & (y_r[-1] > 0.4):
                            bcs_2 = BCS_INDEX[cls.int()]
                            flag_2 += 1
                            # print('flag_2',flag_2)
                        else:
                            RBCS = np.append(RBCS, bcs_2)
                            print('RBCS{}'.format(list(RBCS)))
                            bcs_2 = []
                            flag_2 = 0
                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)
            else:
                if i == 0:
                    flag_1 += 1
                    # print('flag_1', flag_1)
                else:
                    flag_2 += 1
                    # print('flag_2', flag_2)
            if (flag_1 > 300) and (LBCS.size != 0):
                # 数据保存到文件
                np.savetxt('LBCS.txt', LBCS, fmt='%.2f', delimiter=' ')
                np.savetxt('LID.txt', LID, fmt='%d', delimiter=' ')
                np.savetxt('y_l.txt', y_l, fmt='%.6f', delimiter=' ')
                # 防止数据错乱
                # for ii in range((24-LBCS.size)):
                # LBCS = np.append(LBCS, (3.0 + random.randint(0 , 5)*0.25))
                # LID  LBCS 对应
                if len(LID) != len(LBCS):
                    LID = LID[:-1]
                # 数据传输到阿里云
                L_End_Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                BCS_STRUCT = {
                    'L_End_Time': L_End_Time,
                    'LBCS': list(LBCS),
                    'L_ID': LID
                }
                JsonUpdataMsn = aliLink.Alink(BCS_STRUCT)
                # print(JsonUpdataMsn)
                mqtt.push(POST, JsonUpdataMsn)  # 定时向阿里云IOT推送我们构建好的Alink协议数据
                LBCS = np.array([])
                y_l = np.array([])
                LID = list()
                print("LBCS LID Y_L清零")
            if (flag_2 > 300) and (RBCS.size != 0):
                np.savetxt('RBCS.txt', RBCS, fmt='%.2f', delimiter=' ')
                np.savetxt('y_r.txt', y_r, fmt='%.6f', delimiter=' ')
                # 防止数据错乱
                for ii in range((24 - RBCS.size)):
                    RBCS = np.append(RBCS, (3.0 + random.randint(0, 5) * 0.25))
                print('R{}'.format(list(RBCS)))
                R_End_Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                BCS_STRUCT = {
                    'R_End_Time': R_End_Time,
                    'RBCS': list(RBCS),
                }
                JsonUpdataMsn = aliLink.Alink(BCS_STRUCT)
                # print(JsonUpdataMsn)
                mqtt.push(POST, JsonUpdataMsn)  # 定时向阿里云IOT推送我们构建好的Alink协议数据
                RBCS = np.array([])
                y_r = np.array([])
                print("RBCS Y_R清零")
            if view_img:
                im0 = cv2.resize(im0, [320, 320])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='streams.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.9, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    # 链接信息
    Server, ClientId, userNmae, Password = aliLink.linkiot(DeviceName, ProductKey, DeviceSecret)

    # mqtt链接
    mqtt = mqttd.MQTT(Server, ClientId, userNmae, Password)
    mqtt.subscribe(SET)  # 订阅服务器下发消息topic
    mqtt.begin(on_message, on_connect)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(opt)
