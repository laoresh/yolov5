import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# ！！！！！！！！！
# 使用说明：
#     1:参数说明：1）X_in_path:xml文件夹路径
#               2）T_out_path:输出txt文件的路径
#               3）P_out_path:输出标签对应图片的路径存入pic.txt列表（要训练的有效图片的路径）
#               3）classes：类别

X_in_path = r"E:\cow\cow_data_important\old\data128_expend\addNoise\labels"
T_out_path = r"E:\cow\cow_data_important\old\data128_expend\addNoise\labels_txt\train"
P_out_path = r"E:\cow\cow_data_important\old\data128_expend\addNoise\images"
classes = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0",
           "cow:1", "cow:2", "cow:3", "cow:4", "cow:5", "cow:6", "cow:7", "cow:8", "cow:9", "cow:10",
           "cow:11", "cow:12", "cow:13", "cow:14", "cow:15", "cow:16", "cow:17", "cow:18", "cow:19", "cow:20",
           "cow:21", "cow:22", "cow:23", "cow:24", "cow:25", "cow:26", "cow:27", "cow:28", "cow:29", "cow:30",
           "cow:31", "cow:32", "cow:33", "cow:34", "cow:35", "cow:36", "cow:37", "cow:38", "cow:39", "cow:40",
           "cow:41", "cow:42", "cow:43", "cow:44", "cow:45", "cow:46", "cow:47", "cow:48", "cow:49", "cow:50",
           "cow:51", "cow:52", "cow:53", "cow:54", "cow:55", "cow:56", "cow:57", "cow:58", "cow:59", "cow:60",
           "cow:61", "cow:62", "cow:63", "cow:64", "cow:65", "cow:66", "cow:67", "cow:68", "cow:69", "cow:70",
           "cow:71", "cow:72", "cow:73", "cow:74", "cow:75", "cow:76", "cow:77", "cow:78", "cow:79", "cow:80",
           "cow:81", "cow:82", "cow:83", "cow:84", "cow:85", "cow:86", "cow:87", "cow:88", "cow:89", "cow:90",
           "cow:91", "cow:92", "cow:93", "cow:94", "cow:95", "cow:96", "cow:97", "cow:98", "cow:99", "cow:100",
           "cow:101", "cow:102", "cow:103", "cow:104", "cow:105", "cow:106", "cow:107", "cow:108", "cow:109", "cow:110",
           "cow:111", "cow:112", "cow:113", "cow:114", "cow:115"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(in_path, filename, out_path):
    in_file = open(in_path + '/%s.xml' % filename, encoding='UTF-8')
    out_file = open(out_path + '/%s.txt' % filename, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def Processing(X_in_path, T_out_path, P_out_path):
    xml_files = os.listdir(X_in_path)
    list_file = open(P_out_path + '/pic.txt', 'w')
    temp = []
    pic_files = os.listdir(P_out_path)
    for pic in pic_files:
        str = pic.split('.')[0]
        temp.append(str)
    for line in xml_files:
        str1 = line.split('.')[1]
        str2 = line.split('.')[0]
        if str1 != "xml":
            continue
        if str2 not in temp:
            continue
        list_file.write(P_out_path + '\\%s.jpg\n' % (str2))
        convert_annotation(X_in_path, str2, T_out_path)
    list_file.close()


Processing(X_in_path, T_out_path, P_out_path)
