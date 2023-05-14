import cv2


def get_img_from_camera_net(folder_path):
    cap = cv2.VideoCapture("rtsp://admin:daniulive12345@192.168.0.120:554/h265/ch1/main/av_stream")  # 获取网络摄像机

    i = 1
    while i < 3:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        print(str(i))
        cv2.imwrite(folder_path + str(i) + '.jpg', frame)  # 存储为图像
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()


# 测试
if __name__ == '__main__':
    folder_path = 'D:\\camera\\'
    get_img_from_camera_net(folder_path)