#!/usr/bin/python3
import argparse
import time
import torch
import detect
from mqtt import aliLink
from mqtt import mqttd
import os
import numpy as np
from rfid.rfid import *
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
    #     # print(msg.payload)
    #     Msg = json.loads(msg.payload)
    #     switch = Msg['params']['PowerLed']
    #     rpi.powerLed(switch)
    #     print(msg.payload)  # 开关值
    pass


# 连接回调（与阿里云建立链接后的回调函数）
def on_connect(client, userdata, flags, rc):
    pass


def date_utc(time):
    date = time[0:2] + '-' + time[2:4] + '-' + time[4:6] + ' ' + time[9:11] + ':' + time[11:13] + ':' + time[13:15]
    return date

# 链接信息
Server, ClientId, userNmae, Password = aliLink.linkiot(DeviceName, ProductKey, DeviceSecret)

# mqtt链接
mqtt = mqttd.MQTT(Server, ClientId, userNmae, Password)
mqtt.subscribe(SET)  # 订阅服务器下发消息topic
mqtt.begin(on_message, on_connect)
# rfid 
#port = port_init()
#L_ID = read_id(port)
L_ID = list([1001,1002,1003,1004,1005,1006,1007,1008])
print(type(L_ID))
RBCS = [2.5, 4.0, 3.0, 3.75, 3.0, 3.5, 3.0, 3.0, 4.0, 3.5, 3.75, 3.0, 4.0, 3.75, 3.0, 4.0, 3.75, 3.0, 3.5, 3.0, 3.25, 3.25, 3.5, 3.25]
while True:
    time.sleep(10)
    # 构建与云端模型一致的消息结构
    BCS_STRUCT = {
        'R_End_Time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'RBCS': RBCS,
        'L_ID': L_ID
    }
    JsonUpdataMsn = aliLink.Alink(BCS_STRUCT)
    print(JsonUpdataMsn)

    mqtt.push(POST, JsonUpdataMsn)  # 定时向阿里云IOT推送我们构建好的Alink协议数据
