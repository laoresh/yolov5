# coding: utf-8

import numpy as np
import random
import cv2
import glob
import os
import math
import scipy
import xml.etree.cElementTree as ET
import xml.dom.minidom
from xml.dom.minidom import Document
from PIL import Image, ImageDraw
from scipy import misc, ndimage

# 随机平移
def random_translate(img, bboxes, p=0.5):
    # 随机平移
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes


# 随机裁剪
def random_crop(img, bboxes, p=0.5):
    # 随机裁剪
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes


# 随机水平反转
def random_horizontal_flip(img, bboxes, p=0.5):
    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        # bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        # 修改Xmin,Xmax的值
        for bbox in bboxes:
            bbox[0] = w_img - int(bbox[0])
            bbox[2] = w_img - int(bbox[2])

    return img, bboxes


# 随机垂直反转
def random_vertical_flip(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, _, _ = img.shape
        img = img[::-1, :, :]
        # bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]

        # 修改ymin,ymax的值
        for bbox in bboxes:
            bbox[1] = h_img - int(bbox[1])
            bbox[3] = h_img - int(bbox[3])

    return img, bboxes


# 随机顺时针旋转90
def random_rot90_1(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 顺时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 1)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [0, 2]] = h - bboxes[:, [0, 2]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机逆时针旋转
def random_rot90_2(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 逆时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 0)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [1, 3]] = w - bboxes[:, [1, 3]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, bboxes, p=0.5, lower=0.8, upper=1.2):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = img / 255.
    return img, bboxes


# 随机变换通道
def random_swap(im, bboxes, p=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < p:
        swap = perms[random.randrange(0, len(perms))]
        print
        swap
        im[:, :, (0, 1, 2)] = im[:, :, swap]
    return im, bboxes


# 随机变换饱和度
def random_saturation(im, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
    return im, bboxes


# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, p=0.5, delta=18.0):
    if random.random() < p:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0
    return im, bboxes


# 随机旋转0-90角度
def random_rotate_image_func(image):
    # 旋转角度范围
    angle = np.random.uniform(low=0, high=90)
    return scipy.ndimage.interpolation.rotate(image, angle, 'bicubic')


def random_rot(image, bboxes, angle, center=None, scale=1.0, ):
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if bboxes is None:
        for i in range(image.shape[2]):
            image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_CONSTANT)
        return image
    else:
        box_x, box_y, box_label, box_tmp = [], [], [], []
        for box in bboxes:
            box_x.append(int(box[0]))
            box_x.append(int(box[2]))
            box_y.append(int(box[1]))
            box_y.append(int(box[3]))
            box_label.append(box[4])
        box_tmp.append(box_x)
        box_tmp.append(box_y)
        bboxes = np.array(box_tmp)
        ####make it as a 3x3 RT matrix
        full_M = np.row_stack((M, np.asarray([0, 0, 1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        ###make the bboxes as 3xN matrix
        full_bboxes = np.row_stack((bboxes, np.ones(shape=(1, bboxes.shape[1]))))
        bboxes_rotated = np.dot(full_M, full_bboxes)

        bboxes_rotated = bboxes_rotated[0:2, :]
        bboxes_rotated = bboxes_rotated.astype(np.int32)

        result = []
        for i in range(len(box_label)):
            x1, y1, x2, y2 = bboxes_rotated[0][2 * i], bboxes_rotated[1][2 * i], bboxes_rotated[0][2 * i + 1], \
                             bboxes_rotated[1][2 * i + 1]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            x1, x2 = min(w, x1), min(w, x2)
            y1, y2 = min(h, y1), min(h, y2)
            one_box = [x1, y1, x2, y2, box_label[i]]
            result.append(one_box)
        return img_rotated, result


# 平移（需要改变bbox）：平移后的图片需要包含所有的框，否则会对图像的原始标注造成破坏。
def _shift_pic_bboxes(img, bboxes):
    '''
    平移后需要包含所有的框
    参考资料：https://blog.csdn.net/sty945/article/details/79387054
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
                要确保是数值
    输出：
        shift_img：平移后的图像array
        shift_bboxes：平移后的boundingbox的坐标，list
    '''
    # ------------------ 平移图像 ------------------
    w = img.shape[1]
    h = img.shape[0]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(x_max, bbox[3])
        name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离，即每个方向的最大移动距离
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。
    # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动。
    x = int(random.uniform(-(d_to_left / 3), d_to_right / 3))
    y = int(random.uniform(-(d_to_top / 3), d_to_bottom / 3))
    M = np.float32([[1, 0, x], [0, 1, y]])

    # 仿射变换
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))  # 第一个参数表示我们希望进行变换的图片，第二个参数是我们的平移矩阵，第三个希望展示的结果图片的大小

    # ------------------ 平移boundingbox ------------------
    shift_bboxes = list()
    for bbox in bboxes:
        shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y, bbox[4]])

    return shift_img, shift_bboxes


# 裁剪（需要改变bbox）：裁剪后的图片需要包含所有的框，否则会对图像的原始标注造成破坏。
def _crop_img_bboxes(img, bboxes):
    '''
    裁剪后图片要包含所有的框
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
                要确保是数值
    输出：
        crop_img：裁剪后的图像array
        crop_bboxes：裁剪后的boundingbox的坐标，list
    '''
    # ------------------ 裁剪图像 ------------------
    w = img.shape[1]
    h = img.shape[0]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])
        name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离
    d_to_left = x_min
    d_to_right = (w - x_max)
    d_to_top = y_min
    d_to_bottom = (h - y_max)

    # 随机扩展这个最小范围
    crop_x_min = int(x_min - random.uniform(0.7 * d_to_left, d_to_left))  # 修改随机值范围，避免裁的太狠了,这个值可以设(0,1),越大裁剪幅度越小
    crop_y_min = int(y_min - random.uniform(0.7 * d_to_top, d_to_top))  # (0, d_to_top)
    crop_x_max = int(x_max + random.uniform(0.7 * d_to_right, d_to_right))
    crop_y_max = int(y_max + random.uniform(0.7 * d_to_bottom, d_to_bottom))

    # 确保不出界
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)

    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # ------------------ 裁剪bounding boxes ------------------
    crop_bboxes = list()
    for bbox in bboxes:
        crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min,
                            bbox[2] - crop_x_min, bbox[3] - crop_y_min, bbox[4]])

    return crop_img, crop_bboxes


# 改变亮度：改变亮度比较简单，不需要处理bounding boxes
def _changeLight(img, bboxes):
    '''
    adjust_gamma(image, gamma=1, gain=1)函数:
    gamma>1时，输出图像变暗，小于1时，输出图像变亮
    输入：
        img：图像array
    输出：
        img：改变亮度后的图像array,无需修改xml
    '''

    contrast = 1  # 对比度
    # brightness = random.randint(40,80)     #调高亮度，值越大，越亮
    # brightness = random.randint(-60,-20)     #调低亮度，值越低，越暗
    brightness = random.randint(-60, 80)
    adjust_img = cv2.addWeighted(img, contrast, img, 0, brightness)  # cv2.addWeighted(对象,对比度,对象,对比度)

    return adjust_img, bboxes


# 加入噪声：加入噪声也比较简单，不需要处理bounding boxes
def _addNoise(img, bboxes, ):
    '''
    输入：
        img：图像array
    输出：
        img：加入噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
    '''
    # noise_img = random_noise(img, mode='gaussian', clip=True) * 255

    noise_sigma = random.randint(10, 20)  # 生成随机数,这个值越大，噪声越厉害

    temp_image = np.float64(np.copy(img))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image, bboxes


# 旋转：旋转后的图片需要包含所有的框，否则会对图像的原始标注造成破坏。需要注意的是，旋转时图像的一些边角可能会被切除掉，需要避免这种情况。
def _rotate_img_bboxes(img, bboxes, angle=5, scale=1.0):
    '''
    参考：https://blog.csdn.net/saltriver/article/details/79680189
          https://www.ctolib.com/topics-44419.html
    关于仿射变换：https://www.zhihu.com/question/20666664
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    # ---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 获取图像绕着某一点的旋转矩阵
    # getRotationMatrix2D(Point2f center, double angle, double scale)
    # Point2f center：表示旋转的中心点
    # double angle：表示旋转的角度
    # double scale：图像缩放因子
    # 参考：https://cloud.tencent.com/developer/article/1425373
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
    # 新中心点与旧中心点之间的位置
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4)  # ceil向上取整

    # ---------------------- 矫正boundingbox ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        name = bbox[4]
        point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
        point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
        point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))

        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, name])

    return rot_img, rot_bboxes


# 镜像
def _flip_pic_bboxes(img, bboxes):
    '''
    参考：https://blog.csdn.net/jningwei/article/details/78753607
    镜像后的图片要包含所有的框
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        flip_img:镜像后的图像array
        flip_bboxes:镜像后的bounding box的坐标list
    '''
    # ---------------------- 镜像图像 ----------------------
    import copy
    flip_img = copy.deepcopy(img)
    if random.random() < 0.5:
        horizon = True
    else:
        horizon = False
    h, w, _ = img.shape
    if horizon:  # 水平翻转
        flip_img = cv2.flip(flip_img, 1)
    else:
        flip_img = cv2.flip(flip_img, 0)
    # ---------------------- 矫正boundingbox ----------------------
    flip_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        name = bbox[4]
        if horizon:
            flip_bboxes.append([w - x_max, y_min, w - x_min, y_max, name])
        else:
            flip_bboxes.append([x_min, h - y_max, x_max, h - y_min, name])

    return flip_img, flip_bboxes


# 读xml
def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text

        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
        result.append(class_name)  #

        results.append(result)
    return results


# 写xml文件，参数中tmp表示路径，imgname是文件名（没有尾缀）ps有尾缀也无所谓
def writeXml(tmp, imgname, w, h, d, bboxes):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[4]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(bbox[0]))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(bbox[1]))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(bbox[2]))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(bbox[3]))
        ymax.appendChild(ymax_txt)

        print(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])

    xmlname = os.path.splitext(imgname)[0]
    tempfile = tmp + "/%s.xml" % xmlname
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return


if __name__ == "__main__":
    root = r'E:\Shinelon\pycharm2020\pytorch_works\cow_data_important'

    img_dir = root + '/cow_128_images/train'
    print(img_dir)
    anno_path = root + '/cow_128_images/labels-xml/train'
    savepath = root + '/data128_expend'

    # 设置数据扩增的方式
    Method = 'flip'

    # 存储新的anno位置
    anno_new_dir = os.path.join(savepath, Method, 'labels')
    if not os.path.isdir(anno_new_dir):
        os.makedirs(anno_new_dir)

    # 扩增后图片保存的位置
    img_new_dir = os.path.join(savepath, Method, 'images')
    if not os.path.isdir(img_new_dir):
        os.makedirs(img_new_dir)

    img_list = glob.glob("{}/*.jpg".format(img_dir))
    for image_path in img_list:
        img_org = cv2.imread(image_path)
        img = img_org
        file_name = os.path.basename(os.path.splitext(image_path)[0])  # 得到原图的名称
        bboxes = readAnnotations(anno_path + "/" + file_name + ".xml")
        print("img: {},  box: {}".format(image_path, bboxes))

        new_img = img
        new_bboxes = bboxes

        # 选择数据扩增方式

        # if Method == 'random_horizontal_flip':
        #     new_img, new_bboxes = random_vertical_flip(img, np.array(bboxes), 1)

        # if Method == 'random_vertical_flip':
        #     new_img, new_bboxes = random_vertical_flip(img, np.array(bboxes), 1)

        # if Method == 'random_rot90_1':
        #     new_img, new_bboxes = random_rot90_1(img, np.array(bboxes), 1)

        # if Method == 'random_translate':
        #     new_img, new_bboxes = random_translate(img, np.array(bboxes), 1)

        # if Method == 'random_crop':
        #     new_img, new_bboxes = random_crop(img, np.array(bboxes), 1)

        # if Method == 'random_bright':
        #     new_img, new_bboxes = random_bright(img, np.array(bboxes), 1)

        # if Method == 'random_swap':
        #     new_img, new_bboxes = random_swap(img, np.array(bboxes), 1)

        # if Method == 'random_saturation':
        #     new_img, new_bboxes = random_saturation(img, np.array(bboxes), 1)

        # if Method == 'random_hue':
        #     new_img, new_bboxes = random_hue(img, np.array(bboxes), 1)

        if Method == 'shift':  # 平移
            new_img, new_bboxes = _shift_pic_bboxes(img, bboxes)

        if Method == 'crop':  # 裁剪
            new_img, new_bboxes = _crop_img_bboxes(img, bboxes)

        if Method == 'Light':  # 改变亮度
            new_img, new_bboxes = _changeLight(img, bboxes)

        if Method == 'addNoise':  # 加高斯噪声
            new_img, new_bboxes = _addNoise(img, bboxes)

        if Method == 'rotate':  # 旋转
            new_img, new_bboxes = _rotate_img_bboxes(img, bboxes)

        if Method == 'flip':  # 镜像
            new_img, new_bboxes = _flip_pic_bboxes(img, bboxes)

        # 保存新图像
        ext = os.path.splitext(image_path)[-1]  # 得到原图的后缀
        new_img_name = '%s_%s%s' % (file_name, Method, ext)
        w = cv2.imwrite(os.path.join(img_new_dir, new_img_name), new_img)  # 新的命名方式为：原图名称+P+角度

        # 保存新xml文件
        H, W, D = new_img.shape  # 得新图像的高、宽、深度，用于书写xml
        writeXml(anno_new_dir, new_img_name, W, H, D, new_bboxes)

        # img = np.array(img)
        #
        # for box in bboxes:
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.putText(img, str(box[4]), (box[0], max(20, box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #
        # cv2.imshow(image_path, img)
        img_rotate = 0
        # cv2.waitKey(0)
